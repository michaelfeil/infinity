mod dtype;

use hf_hub::api::tokio::{ApiError, ApiRepo};
use infinity_v3_backend_core::{Backend as CoreBackend, Inference};
use std::cmp::{max, min};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{instrument, Span};

#[cfg(any(feature = "libtorch", feature = "oxide"))]
use serde::Deserialize;
#[cfg(any(feature = "libtorch", feature = "oxide"))]
use std::path::Path;

pub use crate::dtype::DType;
pub use infinity_v3_backend_core::{block_attention_config, result_memory};
pub use infinity_v3_backend_core::{
    AudioFeatures, BackendError, Batch, ClefField, DecisionInput, DecisionOutput, Embedding,
    Embeddings, ImagePatches, ModelType, MultimodalEncoding, Pool, Predictions, TokenPredictions,
};

/// A completed result or an owned completion handle after native submission.
/// Deferred results have already been handed to the bounded completion worker,
/// so the model lease can be released before awaiting the result.
pub enum Submission<T> {
    Ready(T, Duration),
    Deferred(oneshot::Receiver<Result<(T, Duration), BackendError>>),
}

impl<T> Submission<T> {
    pub async fn complete(self) -> Result<(T, Duration), BackendError> {
        match self {
            Self::Ready(value, duration) => Ok((value, duration)),
            Self::Deferred(receiver) => receiver.await.map_err(|_| BackendError::Unhealthy)?,
        }
    }
}

/// Select an inference runtime independently of its accelerator installation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum BackendKind {
    Libtorch,
    Oxide,
}

impl Default for BackendKind {
    fn default() -> Self {
        if cfg!(feature = "libtorch") || !cfg!(feature = "oxide") {
            Self::Libtorch
        } else {
            Self::Oxide
        }
    }
}

impl BackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Libtorch => "libtorch",
            Self::Oxide => "oxide",
        }
    }

    pub const fn is_compiled(self) -> bool {
        match self {
            Self::Libtorch => cfg!(feature = "libtorch"),
            Self::Oxide => cfg!(feature = "oxide"),
        }
    }

    pub fn ensure_compiled(self) -> Result<(), BackendError> {
        if self.is_compiled() {
            return Ok(());
        }
        let available = [Self::Libtorch, Self::Oxide]
            .into_iter()
            .filter(|kind| kind.is_compiled())
            .map(Self::as_str)
            .collect::<Vec<_>>();
        Err(BackendError::Start(format!(
            "Backend '{}' was not compiled; compiled backends: {}. Build with --features {}",
            self.as_str(),
            if available.is_empty() {
                "none".into()
            } else {
                available.join(", ")
            },
            match self {
                Self::Libtorch => "libtorch",
                Self::Oxide => "oxide",
            },
        )))
    }

    /// Resolve accelerator selection once before choosing dtype or replica devices.
    pub fn resolve_device(self, torch: TorchDevice) -> Result<TorchDevice, BackendError> {
        match self {
            Self::Libtorch => torch.resolved(),
            Self::Oxide => match torch {
                TorchDevice::Auto | TorchDevice::Cuda => Ok(TorchDevice::Cuda),
                TorchDevice::Cpu => Err(BackendError::Start(
                    "The plain CUDA backend requires a CUDA device; --torch-device cpu is unsupported".into(),
                )),
            },
        }
    }

    /// Whether the selected runtime and accelerator can host GPU replicas.
    pub fn supports_replication(self, torch: TorchDevice) -> bool {
        match self {
            Self::Libtorch => torch == TorchDevice::Cuda,
            Self::Oxide => torch == TorchDevice::Cuda,
        }
    }

    /// Query the selected runtime rather than another backend's CUDA installation.
    pub fn visible_device_count(self, torch: TorchDevice) -> Result<usize, BackendError> {
        match self {
            Self::Libtorch => torch.count(),
            Self::Oxide => {
                self.resolve_device(torch)?;
                #[cfg(feature = "oxide")]
                {
                    infinity_backend_oxide::device_count()
                }
                #[cfg(not(feature = "oxide"))]
                {
                    Err(BackendError::Start(
                        "Plain CUDA was not compiled; build with --features oxide".into(),
                    ))
                }
            }
        }
    }
}

/// LibTorch accelerator family; availability depends on the installed C++ distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum TorchDevice {
    #[default]
    Auto,
    Cpu,
    Cuda,
}
impl TorchDevice {
    pub fn resolved(self) -> Result<Self, BackendError> {
        if self != Self::Auto {
            return Ok(self);
        }
        #[cfg(feature = "libtorch")]
        {
            Ok(
                if infinity_backend_libtorch::device_count("cuda").unwrap_or(0) > 0 {
                    Self::Cuda
                } else {
                    Self::Cpu
                },
            )
        }
        #[cfg(not(feature = "libtorch"))]
        Err(BackendError::Start(
            "LibTorch was not compiled; build with --features libtorch".into(),
        ))
    }
    pub fn count(self) -> Result<usize, BackendError> {
        #[cfg(feature = "libtorch")]
        {
            infinity_backend_libtorch::device_count(&self.name(0))
        }
        #[cfg(not(feature = "libtorch"))]
        Err(BackendError::Start(
            "LibTorch was not compiled; build with --features libtorch".into(),
        ))
    }
    pub fn name(self, index: usize) -> String {
        match self {
            Self::Auto => "auto".into(),
            Self::Cpu => "cpu".into(),
            Self::Cuda => format!("cuda:{index}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Backend {
    /// Channel to communicate with the background thread
    backend_sender: mpsc::Sender<BackendCommand>,
    /// Health status
    health_receiver: watch::Receiver<bool>,
    _backend_thread: Arc<BackendThread>,
    pub radix_mlp_supported: bool,
    pub max_batch_size: Option<usize>,
    pub model_type: ModelType,
}

impl Backend {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        model_path: PathBuf,
        api_repo: Option<ApiRepo>,
        dtype: DType,
        model_type: ModelType,
        dense_path: Option<String>,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        Self::new_shared(
            model_path,
            api_repo.map(Arc::new),
            dtype,
            model_type,
            dense_path,
            device_id,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn new_shared(
        model_path: PathBuf,
        api_repo: Option<Arc<ApiRepo>>,
        dtype: DType,
        model_type: ModelType,
        dense_path: Option<String>,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        Self::new_shared_with_fp8(
            model_path, api_repo, dtype, model_type, dense_path, device_id, false,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn new_shared_with_fp8(
        model_path: PathBuf,
        api_repo: Option<Arc<ApiRepo>>,
        dtype: DType,
        model_type: ModelType,
        dense_path: Option<String>,
        device_id: usize,
        enable_fp8_dynamic: bool,
    ) -> Result<Self, BackendError> {
        Self::new_selected(
            model_path,
            api_repo,
            dtype,
            model_type,
            dense_path,
            device_id,
            enable_fp8_dynamic,
            BackendKind::default(),
            TorchDevice::Auto,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn new_selected(
        model_path: PathBuf,
        api_repo: Option<Arc<ApiRepo>>,
        dtype: DType,
        model_type: ModelType,
        dense_path: Option<String>,
        device_id: usize,
        enable_fp8_dynamic: bool,
        kind: BackendKind,
        torch_device: TorchDevice,
    ) -> Result<Self, BackendError> {
        kind.ensure_compiled()?;
        let (backend_sender, backend_receiver) = mpsc::channel(8);

        let backend = init_backend(
            model_path,
            api_repo,
            dtype,
            model_type.clone(),
            dense_path,
            device_id,
            enable_fp8_dynamic,
            kind,
            torch_device,
        )
        .await?;
        let radix_mlp_supported = backend.supports_radix_mlp();
        let max_batch_size = backend.max_batch_size();

        let (health_sender, health_receiver) = watch::channel(false);
        let _backend_thread =
            Arc::new(BackendThread::new(backend, backend_receiver, health_sender));

        Ok(Self {
            backend_sender,
            health_receiver,
            _backend_thread,
            radix_mlp_supported,
            max_batch_size,
            model_type,
        })
    }

    #[instrument(skip(self))]
    pub async fn warmup(
        &self,
        max_input_length: usize,
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
    ) -> Result<(), BackendError> {
        let warmup_tokens = max_batch_tokens;

        let mut input_ids = Vec::with_capacity(warmup_tokens);
        let mut token_type_ids = Vec::with_capacity(warmup_tokens);
        let mut position_ids = Vec::with_capacity(warmup_tokens);

        let mut cumulative_seq_lengths = vec![0];
        let mut pooled_indices = Vec::new();

        let mut i = 0_u32;
        let mut remaining = warmup_tokens;
        let mut cumulative_length = 0;
        let mut max_length = 0;

        while remaining > 0 {
            let request_length = min(remaining, max_input_length);
            cumulative_length += request_length;
            max_length = max(max_length, request_length as u32);

            input_ids.extend(vec![0; request_length]);
            token_type_ids.extend(vec![0; request_length]);
            position_ids.extend((0..request_length as u32).collect::<Vec<u32>>());

            cumulative_seq_lengths.push(cumulative_length as u32);
            pooled_indices.push(i);

            i += 1;
            remaining = remaining.saturating_sub(max_input_length);
            if let Some(max_batch_requests) = &max_batch_requests {
                if i as usize == *max_batch_requests {
                    break;
                }
            }
        }

        let batch = Batch {
            multimodal: vec![],
            input_ids,
            token_type_ids,
            position_ids,
            cumulative_seq_lengths,
            max_length,
            pooled_indices,
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            fold_gather: None,
            scatter_unfold: None,
            tokens: vec![],
            offsets: vec![],
        };

        self.probe(batch).await
    }

    async fn probe(&self, batch: Batch) -> Result<(), BackendError> {
        let (sender, receiver) = oneshot::channel();
        self.backend_sender
            .send(BackendCommand::Probe(
                batch,
                self.model_type.clone(),
                Span::current(),
                sender,
            ))
            .await
            .map_err(|_| BackendError::Unhealthy)?;
        receiver.await.map_err(|_| BackendError::Unhealthy)?
    }

    #[instrument(skip(self))]
    /// Stop the native model worker and drain deferred completions before returning.
    pub async fn shutdown(&self) {
        let _guard = self._backend_thread.shutdown.lock().await;
        if self._backend_thread.handle.lock().unwrap().is_none() {
            return;
        }
        let _ = self.backend_sender.send(BackendCommand::Shutdown).await;
        let handle = self._backend_thread.handle.lock().unwrap().take();
        if let Some(handle) = handle {
            match tokio::task::spawn_blocking(move || handle.join()).await {
                Ok(Ok(())) => {}
                _ => tracing::error!("Native backend worker failed during shutdown"),
            }
        }
    }

    #[instrument(skip(self))]
    pub async fn health(&self) -> Result<(), BackendError> {
        if *self.health_receiver.borrow() {
            // The backend is healthy. Only do a basic health check by calling the
            // the underlying health method.

            let (sender, receiver) = oneshot::channel();
            self.backend_sender
                .send(BackendCommand::Health(Span::current(), sender))
                .await
                .expect("No backend receiver. This is a bug.");
            receiver.await.expect(
                "Backend blocking task dropped the sender without sending a response. This is a bug.",
            )
        } else {
            // The backend is un-healthy or only just started. Do a more advanced health check
            // by calling the model forward on a test batch

            let batch = Batch {
                multimodal: vec![],
                input_ids: vec![0],
                token_type_ids: vec![0],
                position_ids: vec![0],
                cumulative_seq_lengths: vec![0, 1],
                max_length: 1,
                pooled_indices: vec![0],
                raw_indices: vec![],
                compact_input_ids: None,
                compact_position_ids: None,
                fold_gather: None,
                scatter_unfold: None,
                tokens: vec![],
                offsets: vec![],
            };
            self.probe(batch).await
        }
    }

    #[instrument(skip_all)]
    pub async fn embed(&self, batch: Batch) -> Result<(Embeddings, Duration), BackendError> {
        self.submit_embed(batch).await?.complete().await
    }

    pub async fn submit_embed(&self, batch: Batch) -> Result<Submission<Embeddings>, BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .send(BackendCommand::Embed(batch, Span::current(), sender))
            .await
            .map_err(|_| BackendError::Unhealthy)?;
        receiver.await.map_err(|_| BackendError::Unhealthy)?
    }

    pub async fn decide(
        &self,
        batch: Batch,
        inputs: Vec<DecisionInput>,
    ) -> Result<(Vec<DecisionOutput>, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();
        self.backend_sender
            .send(BackendCommand::Decide(
                batch,
                inputs,
                Span::current(),
                sender,
            ))
            .await
            .map_err(|_| BackendError::Unhealthy)?;
        receiver.await.map_err(|_| BackendError::Unhealthy)?
    }

    #[instrument(skip_all)]
    pub async fn predict(&self, batch: Batch) -> Result<(Predictions, Duration), BackendError> {
        self.submit_predict(batch).await?.complete().await
    }

    pub async fn submit_predict(
        &self,
        batch: Batch,
    ) -> Result<Submission<Predictions>, BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .send(BackendCommand::Predict(batch, Span::current(), sender))
            .await
            .map_err(|_| BackendError::Unhealthy)?;
        receiver.await.map_err(|_| BackendError::Unhealthy)?
    }

    #[instrument(skip_all)]
    pub async fn predict_tokens(
        &self,
        batch: Batch,
    ) -> Result<(TokenPredictions, Duration), BackendError> {
        self.submit_predict_tokens(batch).await?.complete().await
    }

    pub async fn submit_predict_tokens(
        &self,
        batch: Batch,
    ) -> Result<Submission<TokenPredictions>, BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .send(BackendCommand::PredictTokens(
                batch,
                Span::current(),
                sender,
            ))
            .await
            .map_err(|_| BackendError::Unhealthy)?;
        receiver.await.map_err(|_| BackendError::Unhealthy)?
    }
}

#[allow(unused, clippy::too_many_arguments)]
async fn init_backend(
    model_path: PathBuf,
    api_repo: Option<Arc<ApiRepo>>,
    dtype: DType,
    model_type: ModelType,
    dense_path: Option<String>,
    device_id: usize,
    enable_fp8_dynamic: bool,
    kind: BackendKind,
    torch_device: TorchDevice,
) -> Result<Box<dyn CoreBackend + Send>, BackendError> {
    if kind == BackendKind::Libtorch {
        return init_libtorch_backend(
            model_path,
            api_repo,
            dtype,
            model_type,
            dense_path,
            device_id,
            enable_fp8_dynamic,
            torch_device,
        )
        .await;
    }
    kind.resolve_device(torch_device)?;
    #[cfg(feature = "oxide")]
    {
        if enable_fp8_dynamic {
            return Err(BackendError::Start(
                "The plain CUDA backend does not yet support dynamic FP8".into(),
            ));
        }
        let dtype = oxide_dtype(dtype)?;
        if !matches!(
            &model_type,
            ModelType::Embedding(Pool::Cls | Pool::Mean | Pool::LastToken)
        ) {
            return Err(BackendError::Start(
                "The plain CUDA backend currently supports pooled BERT embeddings only".into(),
            ));
        }
        let dense_paths = if let Some(repo) = api_repo.as_ref() {
            // Oxide must distinguish a genuinely absent `modules.json` from a
            // failed fetch: a transient error cannot silently drop a trained
            // projection. The native downloader keeps its existing tolerance.
            let paths = resolve_oxide_dense_modules(repo.as_ref(), dense_path.clone()).await?;
            download_safetensors(Arc::clone(repo))
                .await
                .map_err(|e| BackendError::WeightsNotFound(e.to_string()))?;
            paths
        } else {
            local_dense_paths(&model_path.join("modules.json"), dense_path)
                .map_err(|e| BackendError::Start(e.to_string()))?
        };
        let startup_span = Span::current();
        let model = tokio::task::spawn_blocking(move || {
            startup_span.in_scope(|| {
                infinity_backend_oxide::OxideBackend::new_with_dense_paths(
                    &model_path,
                    dtype,
                    model_type,
                    device_id,
                    dense_paths,
                )
            })
        })
        .await
        .map_err(|e| BackendError::Start(format!("CUDA initialization worker failed: {e}")))??;
        Ok(Box::new(model))
    }
    #[cfg(not(feature = "oxide"))]
    {
        Err(BackendError::Start(
            "Plain CUDA was not compiled; build with --features oxide".into(),
        ))
    }
}

#[allow(unused, clippy::too_many_arguments)]
async fn init_libtorch_backend(
    model_path: PathBuf,
    api_repo: Option<Arc<ApiRepo>>,
    dtype: DType,
    model_type: ModelType,
    dense_path: Option<String>,
    device_id: usize,
    enable_fp8_dynamic: bool,
    torch_device: TorchDevice,
) -> Result<Box<dyn CoreBackend + Send>, BackendError> {
    if model_type == ModelType::Decision {
        if let Some(repo) = api_repo.as_ref() {
            download_decision_sidecars(repo)
                .await
                .map_err(|error| BackendError::WeightsNotFound(error.to_string()))?;
        }
    }
    #[cfg(feature = "libtorch")]
    {
        let dense_paths = if let Some(repo) = api_repo.as_ref() {
            let explicit_dense = dense_path.is_some();
            let paths = download_dense_modules(repo, dense_path)
                .await
                .map_err(|e| BackendError::WeightsNotFound(e.to_string()))?;
            if model_type == ModelType::Decision && model_path.join("vela3v_config.json").exists() {
                // The raw model.safetensors belongs to the span task. Core
                // decisions use this separate, decision-trained backbone.
                for file in ["backbone_h2.safetensors", "decision_adapter.safetensors"] {
                    repo.get(file)
                        .await
                        .map_err(|e| BackendError::WeightsNotFound(e.to_string()))?;
                }
            } else if download_safetensors(repo.clone()).await.is_err() {
                download_file(repo, "pytorch_model.bin")
                    .await
                    .map_err(|e| BackendError::WeightsNotFound(e.to_string()))?;
            }
            // Discovered Dense modules are not an override for classifiers: the
            // backend decides from modules.json whether the checkpoint needs them.
            if model_type == ModelType::Classifier && !explicit_dense {
                None
            } else {
                Some(paths)
            }
        } else {
            local_dense_paths(&model_path.join("modules.json"), dense_path)
                .map_err(|e| BackendError::Start(e.to_string()))?
        };
        let device = torch_device.resolved()?.name(device_id);
        let startup_span = Span::current();
        let model = tokio::task::spawn_blocking(move || {
            startup_span.in_scope(|| {
                infinity_backend_libtorch::LibtorchBackend::new_with_dense_paths_and_fp8(
                    &model_path,
                    &dtype.to_string(),
                    model_type,
                    &device,
                    dense_paths,
                    enable_fp8_dynamic,
                )
            })
        })
        .await
        .map_err(|e| {
            BackendError::Start(format!("LibTorch initialization worker failed: {e}"))
        })??;
        Ok(Box::new(model))
    }
    #[cfg(not(feature = "libtorch"))]
    {
        Err(BackendError::Start(
            "LibTorch was not compiled; build with --features libtorch".into(),
        ))
    }
}

type CompletionJob = Box<dyn FnOnce() + Send>;
struct CompletionThread {
    sender: Option<std::sync::mpsc::SyncSender<CompletionJob>>,
    handle: Option<JoinHandle<()>>,
}
impl CompletionThread {
    fn new() -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<CompletionJob>(2);
        let handle = std::thread::Builder::new()
            .name("tei-completion".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    job();
                }
            })
            .expect("Could not start completion worker");
        Self {
            sender: Some(sender),
            handle: Some(handle),
        }
    }
}
impl Drop for CompletionThread {
    fn drop(&mut self) {
        self.sender.take();
        let _ = self.handle.take().unwrap().join();
    }
}

fn deliver<T: Send + 'static>(
    result: Result<Inference<T>, BackendError>,
    sender: oneshot::Sender<Result<Submission<T>, BackendError>>,
    start: Instant,
    span: Span,
    health: &watch::Sender<bool>,
    completion: &mut Option<CompletionThread>,
) {
    match result {
        Ok(Inference::Deferred(complete)) => {
            let health = health.clone();
            let (response, receiver) = oneshot::channel();
            let job = Box::new(move || {
                let _span = span.entered();
                // Complete even after receiver cancellation: retain pins/slots until
                // the GPU finishes, and never free native work on a Tokio worker.
                let result = complete().map(|value| (value, start.elapsed()));
                let _ = health.send(
                    result.is_ok()
                        || result
                            .as_ref()
                            .is_err_and(|e| matches!(e, BackendError::Overloaded(_))),
                );
                let _ = response.send(result);
            });
            completion
                .get_or_insert_with(CompletionThread::new)
                .sender
                .as_ref()
                .unwrap()
                .send(job)
                .expect("Completion worker exited");
            // Return the completion handle only after native submission and
            // bounded handoff. Dropping it still leaves the worker owning the job.
            let _ = sender.send(Ok(Submission::Deferred(receiver)));
        }
        result => {
            let result = result
                .and_then(Inference::complete)
                .map(|value| Submission::Ready(value, start.elapsed()));
            let _ = health.send(
                result.is_ok()
                    || result
                        .as_ref()
                        .is_err_and(|e| matches!(e, BackendError::Overloaded(_))),
            );
            let _ = sender.send(result);
        }
    }
}

#[derive(Debug)]
struct BackendThread {
    handle: std::sync::Mutex<Option<JoinHandle<()>>>,
    shutdown: tokio::sync::Mutex<()>,
}

impl BackendThread {
    fn new(
        backend: Box<dyn CoreBackend + Send>,
        mut backend_receiver: mpsc::Receiver<BackendCommand>,
        health_sender: watch::Sender<bool>,
    ) -> Self {
        let handle = std::thread::spawn(move || {
            let mut completion = None;
            while let Some(cmd) = backend_receiver.blocking_recv() {
                let start = Instant::now();
                match cmd {
                    BackendCommand::Shutdown => break,
                    BackendCommand::Probe(batch, model_type, span, sender) => {
                        let _span = span.entered();
                        let result = backend.probe(batch, &model_type);
                        let _ = health_sender.send(
                            result.is_ok() || matches!(&result, Err(BackendError::Overloaded(_))),
                        );
                        let _ = sender.send(result);
                    }
                    BackendCommand::Decide(batch, inputs, span, sender) => {
                        let _span = span.entered();
                        let result = backend.decide(batch, inputs);
                        let _ = health_sender.send(
                            result.is_ok() || matches!(&result, Err(BackendError::Overloaded(_))),
                        );
                        let _ = sender.send(result.map(|output| (output, start.elapsed())));
                    }
                    BackendCommand::Health(span, sender) => {
                        let _span = span.entered();
                        let result = backend.health();
                        let _ = health_sender.send(result.is_ok());
                        let _ = sender.send(result);
                    }
                    BackendCommand::Embed(batch, span, sender) => {
                        let result = {
                            let _span = span.enter();
                            backend.submit_embed(batch)
                        };
                        deliver(result, sender, start, span, &health_sender, &mut completion);
                    }
                    BackendCommand::Predict(batch, span, sender) => {
                        let result = {
                            let _span = span.enter();
                            backend.submit_predict(batch)
                        };
                        deliver(result, sender, start, span, &health_sender, &mut completion);
                    }
                    BackendCommand::PredictTokens(batch, span, sender) => {
                        let result = {
                            let _span = span.enter();
                            backend.submit_predict_tokens(batch)
                        };
                        deliver(result, sender, start, span, &health_sender, &mut completion);
                    }
                };
            }
            // Join/drain completion before destroying the model and its graph cache.
            drop(completion);
        });
        Self {
            handle: std::sync::Mutex::new(Some(handle)),
            shutdown: tokio::sync::Mutex::new(()),
        }
    }
}

impl Drop for BackendThread {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.get_mut().unwrap().take() {
            if handle.join().is_err() {
                tracing::error!("Native backend worker panicked");
            }
        }
    }
}

enum BackendCommand {
    Shutdown,
    Probe(
        Batch,
        ModelType,
        Span,
        oneshot::Sender<Result<(), BackendError>>,
    ),
    Decide(
        Batch,
        Vec<DecisionInput>,
        Span,
        oneshot::Sender<Result<(Vec<DecisionOutput>, Duration), BackendError>>,
    ),
    Health(Span, oneshot::Sender<Result<(), BackendError>>),
    Embed(
        Batch,
        Span,
        oneshot::Sender<Result<Submission<Embeddings>, BackendError>>,
    ),
    Predict(
        Batch,
        Span,
        #[allow(clippy::type_complexity)]
        oneshot::Sender<Result<Submission<Predictions>, BackendError>>,
    ),
    PredictTokens(
        Batch,
        Span,
        #[allow(clippy::type_complexity)]
        oneshot::Sender<Result<Submission<TokenPredictions>, BackendError>>,
    ),
}

async fn download_decision_sidecars(api: &ApiRepo) -> Result<(), ApiError> {
    // These sidecars select optional decision heads. Missing cached probes
    // must also be tolerated offline; a discovered head's companion is required.
    if api.get("rl_agent_config.json").await.is_ok() {
        api.get("encoder/config.json").await?;
    } else if api.get("decision_config.json").await.is_ok() {
        api.get("readout.safetensors").await?;
    } else if api.get("joint_head_config.json").await.is_ok() {
        api.get("joint_head.safetensors").await?;
    }
    Ok(())
}

async fn download_safetensors(api: Arc<ApiRepo>) -> Result<Vec<PathBuf>, ApiError> {
    // Single file
    tracing::info!("Downloading `model.safetensors`");
    match api.get("model.safetensors").await {
        Ok(p) => return Ok(vec![p]),
        Err(err) => tracing::warn!("Could not download `model.safetensors`: {}", err),
    };

    // Sharded weights
    // Download and parse index file
    tracing::info!("Downloading `model.safetensors.index.json`");
    let index_file = api.get("model.safetensors.index.json").await?;
    let index_file_string = std::fs::read_to_string(index_file)?;
    let safetensors_filenames = safetensor_shards(&index_file_string)?;

    // Download weight files
    let handles: Vec<_> = safetensors_filenames
        .into_iter()
        .map(|n| {
            let api = Arc::clone(&api);
            tokio::spawn(async move {
                tracing::info!("Downloading `{}`", n);
                api.get(&n).await
            })
        })
        .collect();

    let mut safetensors_files = Vec::with_capacity(handles.len());
    for handle in handles {
        // Await the JoinHandle to get the result of the task,
        // then unpack the inner result from api.get()
        safetensors_files.push(handle.await??);
    }

    Ok(safetensors_files)
}

fn safetensor_shards(index: &str) -> Result<Vec<String>, std::io::Error> {
    use std::{collections::BTreeSet, io, path::Component};
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid safetensors weight index",
        )
    };
    let json: serde_json::Value = serde_json::from_str(index).map_err(|_| invalid())?;
    let map = json
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .filter(|map| !map.is_empty())
        .ok_or_else(invalid)?;
    let mut shards = BTreeSet::new();
    for value in map.values() {
        let file = value.as_str().ok_or_else(invalid)?;
        if !file.ends_with(".safetensors")
            || file.contains(['\\', '\0'])
            || !std::path::Path::new(file)
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
        {
            return Err(invalid());
        }
        shards.insert(file.to_owned());
    }
    Ok(shards.into_iter().collect())
}

#[cfg(test)]
mod startup_tests {
    use super::*;

    #[test]
    fn invalid_weight_indexes_fail_without_panicking_or_following_parent_paths() {
        for index in [
            "not JSON",
            r#"{}"#,
            r#"{"weight_map":{}}"#,
            r#"{"weight_map":{"a":3}}"#,
            r#"{"weight_map":{"a":"../outside.safetensors"}}"#,
            r#"{"weight_map":{"a":"/outside.safetensors"}}"#,
            r#"{"weight_map":{"a":"weights.bin"}}"#,
        ] {
            assert!(safetensor_shards(index).is_err(), "{index}");
        }
        assert_eq!(
            safetensor_shards(
                r#"{"weight_map":{"a":"b.safetensors","b":"a.safetensors","c":"b.safetensors"}}"#
            )
            .unwrap(),
            ["a.safetensors", "b.safetensors"]
        );
    }

    #[cfg(feature = "oxide")]
    #[test]
    fn oxide_dtype_factory_defaults_to_bf16_and_admits_explicit_f16() {
        assert_eq!(oxide_dtype(DType::Auto).unwrap(), "bfloat16");
        assert_eq!(oxide_dtype(DType::Bfloat16).unwrap(), "bfloat16");
        assert_eq!(oxide_dtype(DType::Float16).unwrap(), "float16");
        let error = oxide_dtype(DType::Float32).unwrap_err().to_string();
        assert!(error.contains("bfloat16 or float16"), "{error}");
    }

    #[cfg(feature = "oxide")]
    #[test]
    fn oxide_rejects_unsupported_options_before_loading_checkpoint_or_driver() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        for (dtype, model_type, dense, fp8, device, diagnostic) in [
            (
                DType::Bfloat16,
                ModelType::Embedding(Pool::Mean),
                None,
                false,
                TorchDevice::Cpu,
                "requires a CUDA device",
            ),
            (
                DType::Bfloat16,
                ModelType::Embedding(Pool::Mean),
                None,
                true,
                TorchDevice::Cuda,
                "dynamic FP8",
            ),
            (
                DType::Float32,
                ModelType::Embedding(Pool::Mean),
                None,
                false,
                TorchDevice::Cuda,
                "bfloat16 or float16",
            ),
            (
                DType::Bfloat16,
                ModelType::Classifier,
                None,
                false,
                TorchDevice::Cuda,
                "pooled BERT embeddings",
            ),
        ] {
            let result = runtime.block_on(init_backend(
                PathBuf::from("missing-checkpoint"),
                None,
                dtype,
                model_type,
                dense,
                0,
                fp8,
                BackendKind::Oxide,
                device,
            ));
            let error = result.err().expect("unsupported options must fail");
            assert!(matches!(error, BackendError::Start(_)));
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
    }
}

#[cfg(any(feature = "libtorch", feature = "oxide"))]
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize, PartialEq)]
enum ModuleType {
    // Output quantization is omitted when serving floating-point embeddings.
    #[serde(rename = "st_quantize.FlexibleQuantizer")]
    FlexibleQuantizer,
    #[serde(
        rename = "sentence_transformers.multi_vector_encoder.modules.multi_vector_mask.MultiVectorMask"
    )]
    MultiVectorMask,
    #[serde(
        rename = "sentence_transformers.models.Dense",
        alias = "sentence_transformers.base.modules.dense.Dense"
    )]
    Dense,
    #[serde(
        rename = "sentence_transformers.models.Normalize",
        alias = "sentence_transformers.base.modules.normalize.Normalize"
    )]
    Normalize,
    #[serde(
        rename = "sentence_transformers.models.Pooling",
        alias = "sentence_transformers.sentence_transformer.modules.pooling.Pooling"
    )]
    Pooling,
    #[serde(
        rename = "sentence_transformers.models.Transformer",
        alias = "sentence_transformers.base.modules.transformer.Transformer"
    )]
    Transformer,
}

#[cfg(any(feature = "libtorch", feature = "oxide"))]
#[derive(Debug, Clone, Deserialize)]
struct ModuleConfig {
    #[allow(dead_code)]
    #[serde(default)]
    idx: usize,
    #[allow(dead_code)]
    #[serde(default)]
    name: String,
    path: String,
    #[serde(rename = "type")]
    module_type: ModuleType,
}

#[cfg(any(feature = "libtorch", feature = "oxide"))]
async fn download_file(api: &ApiRepo, file_path: &str) -> Result<PathBuf, ApiError> {
    tracing::info!("Downloading `{}`", file_path);
    api.get(file_path).await
}

#[cfg(any(feature = "libtorch", feature = "oxide"))]
fn parse_dense_paths_from_modules_sync(modules_path: &Path) -> Result<Vec<String>, std::io::Error> {
    let content = std::fs::read_to_string(modules_path)?;
    let modules: Vec<ModuleConfig> = serde_json::from_str(&content)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;

    Ok(modules
        .into_iter()
        .filter(|module| module.module_type == ModuleType::Dense)
        .map(|module| module.path)
        .collect::<Vec<String>>())
}

#[cfg(any(feature = "libtorch", feature = "oxide"))]
async fn parse_dense_paths_from_modules(
    modules_path: &Path,
) -> Result<Vec<String>, std::io::Error> {
    parse_dense_paths_from_modules_sync(modules_path)
}

/// Cold mapping from the public dtype selector to the oxide backend's storage
/// name. Auto keeps the existing BF16 default; FP16 is explicit.
#[cfg(feature = "oxide")]
fn oxide_dtype(dtype: DType) -> Result<&'static str, BackendError> {
    match dtype {
        DType::Auto | DType::Bfloat16 => Ok("bfloat16"),
        DType::Float16 => Ok("float16"),
        DType::Float32 => Err(BackendError::Start(
            "The oxide backend supports --dtype bfloat16 or float16".into(),
        )),
    }
}

/// Result of fetching the optional Sentence Transformers `modules.json`.
#[cfg(feature = "oxide")]
#[derive(Debug, Clone, PartialEq, Eq)]
enum ModulesFetch {
    /// The checkpoint genuinely has no `modules.json` (HTTP 404).
    Absent,
    /// The file was fetched successfully.
    Contents(String),
    /// Fetching or reading the file failed for any reason other than absence.
    Failed(String),
}

/// Classify a fetch status. Only a genuine 404 means the optional metadata is
/// absent; every other failure must stop startup instead of dropping Dense.
#[cfg(feature = "oxide")]
fn classify_modules_status(status: Option<u16>, message: String) -> ModulesFetch {
    match status {
        Some(404) => ModulesFetch::Absent,
        Some(status) => ModulesFetch::Failed(format!("HTTP {status}: {message}")),
        None => ModulesFetch::Failed(message),
    }
}

#[cfg(feature = "oxide")]
fn api_error_status(error: &ApiError) -> Option<u16> {
    match error {
        ApiError::RequestError(error) => error.status().map(|status| status.as_u16()),
        ApiError::TooManyRetries(error) => api_error_status(error),
        _ => None,
    }
}

/// Metadata entry owned by the Oxide discovery path. Unlike `ModuleConfig`,
/// non-Dense modules may omit `path` (notably a valid root Transformer), while
/// a Dense entry without a path is rejected before any model starts.
#[cfg(feature = "oxide")]
#[derive(Debug, Deserialize)]
struct OxideModuleMetadata {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    path: Option<String>,
}

#[cfg(feature = "oxide")]
fn is_oxide_dense_module_type(kind: &str) -> bool {
    kind.starts_with("sentence_transformers.") && kind.rsplit('.').next() == Some("Dense")
}

#[cfg(feature = "oxide")]
fn validate_oxide_dense_path(path: &str) -> Result<(), String> {
    use std::path::Component;
    if path.is_empty()
        || path.contains('\\')
        || !Path::new(path)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
    {
        return Err(
            "Dense module path must be a nonempty relative path inside the model directory".into(),
        );
    }
    Ok(())
}

#[cfg(feature = "oxide")]
fn oxide_declared_dense_paths(contents: &str) -> Result<Vec<String>, String> {
    let modules: Vec<OxideModuleMetadata> = serde_json::from_str(contents)
        .map_err(|error| format!("malformed modules.json: {error}"))?;
    let mut paths = Vec::new();
    for module in modules {
        if !is_oxide_dense_module_type(&module.kind) {
            continue;
        }
        let path = module
            .path
            .ok_or_else(|| "Dense module metadata is missing its path".to_string())?;
        validate_oxide_dense_path(&path)?;
        paths.push(path);
    }
    Ok(paths)
}

/// Resolve the ordered Dense path list from a strict metadata fetch plus the
/// explicit `--dense-path` policy: a single discovered module may be
/// overridden, multiple modules keep authoritative metadata order, and any
/// absence/zero-Dense result uses the explicit override when present.
#[cfg(feature = "oxide")]
fn oxide_dense_paths(
    fetched: ModulesFetch,
    override_path: Option<String>,
) -> Result<Option<Vec<String>>, String> {
    let declared = match fetched {
        ModulesFetch::Absent => Vec::new(),
        ModulesFetch::Contents(contents) => oxide_declared_dense_paths(&contents)?,
        ModulesFetch::Failed(error) => {
            return Err(format!("could not fetch modules.json: {error}"));
        }
    };
    if declared.len() <= 1 {
        if let Some(path) = override_path.as_deref() {
            validate_oxide_dense_path(path)?;
        }
    }
    match declared.len() {
        0 => Ok(override_path.map(|path| vec![path])),
        1 => {
            let Some(discovered) = declared.into_iter().next() else {
                return Err("Dense module metadata disappeared during resolution".into());
            };
            Ok(Some(vec![override_path.unwrap_or(discovered)]))
        }
        _ => Ok(Some(declared)),
    }
}

#[cfg(feature = "oxide")]
trait DenseModuleSource {
    type ModulesFuture<'a>: std::future::Future<Output = ModulesFetch> + Send + 'a
    where
        Self: 'a;
    type DownloadFuture<'a>: std::future::Future<Output = Result<(), String>> + Send + 'a
    where
        Self: 'a;

    fn fetch_modules(&self) -> Self::ModulesFuture<'_>;
    fn download_module<'a>(&'a self, path: &'a str) -> Self::DownloadFuture<'a>;
}

#[cfg(feature = "oxide")]
impl DenseModuleSource for ApiRepo {
    type ModulesFuture<'a>
        = std::pin::Pin<Box<dyn std::future::Future<Output = ModulesFetch> + Send + 'a>>
    where
        Self: 'a;
    type DownloadFuture<'a>
        = std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>>
    where
        Self: 'a;

    fn fetch_modules(&self) -> Self::ModulesFuture<'_> {
        Box::pin(async move {
            match download_file(self, "modules.json").await {
                Ok(path) => match std::fs::read_to_string(&path) {
                    Ok(contents) => ModulesFetch::Contents(contents),
                    Err(error) => ModulesFetch::Failed(format!(
                        "could not read `{}`: {error}",
                        path.display()
                    )),
                },
                Err(error) => classify_modules_status(api_error_status(&error), error.to_string()),
            }
        })
    }

    fn download_module<'a>(&'a self, path: &'a str) -> Self::DownloadFuture<'a> {
        Box::pin(async move {
            download_dense_module(self, path)
                .await
                .map(|_| ())
                .map_err(|error| error.to_string())
        })
    }
}

/// Strict Oxide remote Dense discovery. Metadata failures and required module
/// download failures surface as typed startup errors; only genuine absence or
/// a metadata file with zero Dense modules can resolve to no projection.
#[cfg(feature = "oxide")]
async fn resolve_oxide_dense_modules<S: DenseModuleSource + Sync>(
    source: &S,
    override_path: Option<String>,
) -> Result<Option<Vec<String>>, BackendError> {
    let paths = oxide_dense_paths(source.fetch_modules().await, override_path)
        .map_err(|error| BackendError::Start(format!("Oxide Dense discovery failed: {error}")))?;
    if let Some(paths) = &paths {
        for path in paths {
            source.download_module(path).await.map_err(|error| {
                BackendError::Start(format!(
                    "Oxide Dense module `{path}` could not be downloaded: {error}"
                ))
            })?;
        }
    }
    Ok(paths)
}

/// Local `--dense-path` policy shared with the native loader: one discovered
/// Dense module may be overridden, while multiple sequential modules must all
/// come from `modules.json`. With no Sentence Transformers metadata, the
/// explicit path is used as the only module.
#[cfg(any(feature = "libtorch", feature = "oxide"))]
fn local_dense_paths(
    modules_path: &Path,
    override_path: Option<String>,
) -> Result<Option<Vec<String>>, std::io::Error> {
    let Some(override_path) = override_path else {
        return Ok(None);
    };
    if modules_path.exists() {
        let paths = parse_dense_paths_from_modules_sync(modules_path)?;
        if paths.len() > 1 {
            Ok(Some(paths))
        } else {
            Ok(Some(vec![override_path]))
        }
    } else {
        Ok(Some(vec![override_path]))
    }
}

#[cfg(any(feature = "libtorch", feature = "oxide"))]
#[instrument(skip_all)]
pub async fn download_dense_modules(
    api: &ApiRepo,
    dense_path: Option<String>,
) -> Result<Vec<String>, ApiError> {
    match download_file(api, "modules.json").await {
        Ok(modules_path) => {
            // If `modules.json` exists, then parse it to capture the Dense modules
            match parse_dense_paths_from_modules(&modules_path).await {
                Ok(module_paths) => {
                    match module_paths.len() {
                        0 => Ok(vec![]),
                        // NOTE: if there's only a single Dense module defined i.e., there are
                        // no sequential Dense modules to be applied, then the one defined in
                        // `modules.json` will be downloaded, unless the user has specified
                        // another valid `--dense-path` (that exists in the repository), e.g.
                        // defualt might be set to `2_Dense_1024/`, but the user might want to
                        // use `2_Dense/` instead (see https://huggingface.co/NovaSearch/stella_en_400M_v5)
                        1 => {
                            let path_to_use = if let Some(ref user_path) = dense_path {
                                if user_path != &module_paths[0] {
                                    tracing::info!("`{}` found in `modules.json`, but using provided `--dense-path={user_path}` instead", module_paths[0]);
                                }
                                user_path.clone()
                            } else {
                                module_paths[0].clone()
                            };

                            download_dense_module(api, &path_to_use)
                                .await
                                .map_err(|err| {
                                    tracing::error!(
                                        "Failed to download dense module {}: {}",
                                        path_to_use,
                                        err
                                    );
                                    err
                                })?;
                            Ok(vec![path_to_use])
                        }
                        // NOTE: in any other case i.e., more than 1 Dense module, then download
                        // them all, and then sort them to ensure those are applied sequentially
                        _ => {
                            if dense_path.is_some() {
                                tracing::warn!("A value for `--dense-path` was provided, but since there's more than one subsequent Dense module, then the provided value will be ignored.");
                            }

                            for module_path in &module_paths {
                                // NOTE: since the Dense modules here are specified in the
                                // `modules.json` file, then fail if any of those cannot be
                                // downloaded
                                download_dense_module(api, module_path)
                                    .await
                                    .map_err(|err| {
                                        tracing::error!(
                                            "Failed to download `{module_path}` file: {err}"
                                        );
                                        err
                                    })?;
                            }
                            Ok(module_paths)
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!("`modules.json` could be downloaded but parsing the modules failed: {err}; so no Dense modules will be downloaded.");
                    Ok(vec![])
                }
            }
        }
        // NOTE: if `modules.json` is not there, then no modules will be downloaded, which most
        // likely means that the model is not a Sentence Transformer model
        // Offline caches may report a connection error rather than a 404 for
        // this optional file. Once found, required Dense downloads above still
        // propagate their errors.
        Err(_) => Ok(vec![]),
    }
}

#[cfg(any(feature = "libtorch", feature = "oxide"))]
async fn download_dense_module(api: &ApiRepo, dense_path: &str) -> Result<PathBuf, ApiError> {
    // Download `config.json` for the Dense module
    let config_file = format!("{}/config.json", dense_path);
    let config_path = match download_file(api, &config_file).await {
        Ok(path) => path,
        Err(err) => {
            tracing::warn!("Failed to download `{config_file}` file: {err}");
            return Err(err);
        }
    };

    // Try to download the `model.safetensors` first
    let safetensors_file = format!("{}/model.safetensors", dense_path);
    if let Err(err) = download_file(api, &safetensors_file).await {
        tracing::warn!("Failed to download `{safetensors_file}` file: {err}");
        // Fallback to former `pytorch_model.bin`
        let pytorch_file = format!("{}/pytorch_model.bin", dense_path);
        if let Err(err) = download_file(api, &pytorch_file).await {
            tracing::warn!("Failed to download `{pytorch_file}` file: {err}");
            return Err(err);
        }
    }

    Ok(config_path.parent().unwrap().to_path_buf())
}

#[cfg(test)]
mod completion_tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    };

    struct DeferredBackend {
        gate: Arc<Mutex<std::sync::mpsc::Receiver<Result<(), BackendError>>>>,
        started: std::sync::mpsc::Sender<()>,
        completed: Arc<AtomicUsize>,
        dropped: Arc<AtomicUsize>,
        probes: Arc<AtomicUsize>,
    }
    impl Drop for DeferredBackend {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl CoreBackend for DeferredBackend {
        fn probe(&self, _: Batch, model_type: &ModelType) -> Result<(), BackendError> {
            assert!(matches!(model_type, ModelType::Embedding(_)));
            self.probes.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn health(&self) -> Result<(), BackendError> {
            Ok(())
        }
        fn embed(&self, _: Batch) -> Result<Embeddings, BackendError> {
            unreachable!()
        }
        fn predict(&self, _: Batch) -> Result<Predictions, BackendError> {
            unreachable!()
        }
        fn predict_tokens(&self, _: Batch) -> Result<TokenPredictions, BackendError> {
            unreachable!()
        }
        fn submit_embed(&self, _: Batch) -> Result<Inference<Embeddings>, BackendError> {
            let (gate, started, completed) = (
                self.gate.clone(),
                self.started.clone(),
                self.completed.clone(),
            );
            Ok(Inference::Deferred(Box::new(move || {
                assert_eq!(std::thread::current().name(), Some("tei-completion"));
                started.send(()).unwrap();
                let result = gate.lock().unwrap().recv().unwrap();
                completed.fetch_add(1, Ordering::SeqCst);
                result.map(|_| Embeddings::default())
            })))
        }
    }
    fn batch() -> Batch {
        Batch {
            multimodal: vec![],
            input_ids: vec![1],
            token_type_ids: vec![0],
            position_ids: vec![0],
            cumulative_seq_lengths: vec![0, 1],
            max_length: 1,
            pooled_indices: vec![0],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        }
    }
    struct AdmittedBackend(Arc<result_memory::ResultPool>);
    impl CoreBackend for AdmittedBackend {
        fn health(&self) -> Result<(), BackendError> {
            Ok(())
        }
        fn embed(&self, _: Batch) -> Result<Embeddings, BackendError> {
            let memory = self.0.reserve(64)?;
            let mut result = Embeddings::default();
            result.memory = Some(memory);
            result.insert(0, Embedding::Pooled(vec![1.0; 16]));
            Ok(result)
        }
        fn predict(&self, _: Batch) -> Result<Predictions, BackendError> {
            unreachable!()
        }
        fn predict_tokens(&self, _: Batch) -> Result<TokenPredictions, BackendError> {
            unreachable!()
        }
        fn decide(
            &self,
            _: Batch,
            _: Vec<DecisionInput>,
        ) -> Result<Vec<DecisionOutput>, BackendError> {
            let memory = self.0.reserve(64)?;
            Ok(vec![DecisionOutput {
                logits: vec![1.0; 16],
                action_probability: 0.5,
                memory: Some(memory),
            }])
        }
    }
    #[tokio::test]
    async fn serving_worker_recovers_after_result_admission_rejection() {
        let pool = result_memory::ResultPool::new(64).unwrap();
        let held = pool.reserve(64).unwrap();
        let (sender, receiver) = mpsc::channel(8);
        let (health, health_receiver) = watch::channel(true);
        let worker = BackendThread::new(Box::new(AdmittedBackend(pool.clone())), receiver, health);
        let backend = Backend {
            backend_sender: sender,
            health_receiver,
            _backend_thread: Arc::new(worker),
            radix_mlp_supported: false,
            max_batch_size: Some(1),
            model_type: ModelType::Embedding(Pool::Mean),
        };
        assert!(matches!(
            backend.embed(batch()).await,
            Err(BackendError::Overloaded(_))
        ));
        assert!(*backend.health_receiver.borrow());
        drop(held);
        let (result, _) = backend.embed(batch()).await.unwrap();
        assert!(matches!(&result[&0],Embedding::Pooled(values) if values==&vec![1.0;16]));
        assert!(*backend.health_receiver.borrow());
        drop(result);
        assert_eq!(pool.used(), 0);
        // Decision requests use the synchronous worker path, so verify its
        // admission failures preserve health and release ownership too.
        let held = pool.reserve(64).unwrap();
        assert!(matches!(
            backend.decide(batch(), vec![DecisionInput::Warmup]).await,
            Err(BackendError::Overloaded(_))
        ));
        assert!(*backend.health_receiver.borrow());
        drop(held);
        let (result, _) = backend
            .decide(batch(), vec![DecisionInput::Warmup])
            .await
            .unwrap();
        assert_eq!(result[0].logits, vec![1.0; 16]);
        assert!(*backend.health_receiver.borrow());
        drop(result);
        assert_eq!(pool.used(), 0);
    }
    #[tokio::test]
    async fn overloaded_probe_preserves_health_and_recovers() {
        for model_type in [ModelType::Embedding(Pool::Mean), ModelType::Decision] {
            let pool = result_memory::ResultPool::new(64).unwrap();
            let held = pool.reserve(64).unwrap();
            let (sender, receiver) = mpsc::channel(8);
            let (health, health_receiver) = watch::channel(true);
            let worker =
                BackendThread::new(Box::new(AdmittedBackend(pool.clone())), receiver, health);
            let backend = Backend {
                backend_sender: sender,
                health_receiver,
                _backend_thread: Arc::new(worker),
                radix_mlp_supported: false,
                max_batch_size: Some(1),
                model_type,
            };
            assert!(matches!(
                backend.probe(batch()).await,
                Err(BackendError::Overloaded(_))
            ));
            assert!(*backend.health_receiver.borrow());
            // Health must use the lightweight check even while the result pool
            // is still full; a second inference probe would overload again.
            backend.health().await.unwrap();
            assert_eq!(pool.used(), 64);
            drop(held);
            backend.probe(batch()).await.unwrap();
            assert!(*backend.health_receiver.borrow());
            assert_eq!(pool.used(), 0);
        }
    }

    #[test]
    fn deferred_completion_does_not_block_model_and_drains_after_cancellation() {
        let (gate, wait) = std::sync::mpsc::channel();
        let (started, start) = std::sync::mpsc::channel();
        let completed = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let probes = Arc::new(AtomicUsize::new(0));
        let backend = Box::new(DeferredBackend {
            gate: Arc::new(Mutex::new(wait)),
            started,
            completed: completed.clone(),
            dropped: dropped.clone(),
            probes: probes.clone(),
        });
        let (commands, receiver) = mpsc::channel(8);
        let (health, _) = watch::channel(false);
        let worker = BackendThread::new(backend, receiver, health);
        let (response, canceled) = oneshot::channel();
        commands
            .blocking_send(BackendCommand::Embed(batch(), Span::none(), response))
            .unwrap();
        let canceled = canceled.blocking_recv().unwrap().unwrap();
        assert!(matches!(canceled, Submission::Deferred(_)));
        start.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(completed.load(Ordering::SeqCst), 0);
        drop(canceled);
        let (response, probed) = oneshot::channel();
        commands
            .blocking_send(BackendCommand::Probe(
                batch(),
                ModelType::Embedding(Pool::Cls),
                Span::none(),
                response,
            ))
            .unwrap();
        probed.blocking_recv().unwrap().unwrap();
        assert_eq!(probes.load(Ordering::SeqCst), 1);
        assert_eq!(completed.load(Ordering::SeqCst), 0);
        let (response, result) = oneshot::channel();
        commands
            .blocking_send(BackendCommand::Embed(batch(), Span::none(), response))
            .unwrap();
        let result = match result.blocking_recv().unwrap().unwrap() {
            Submission::Deferred(result) => result,
            Submission::Ready(..) => panic!("expected deferred completion"),
        };
        // Cancellation before the submission response must also leave the job
        // owned by the worker, alongside the two existing completions.
        let (response, canceled) = oneshot::channel();
        drop(canceled);
        commands
            .blocking_send(BackendCommand::Embed(batch(), Span::none(), response))
            .unwrap();
        let (response, healthy) = oneshot::channel();
        commands
            .blocking_send(BackendCommand::Health(Span::none(), response))
            .unwrap();
        healthy.blocking_recv().unwrap().unwrap();
        gate.send(Ok(())).unwrap();
        start.recv_timeout(Duration::from_secs(5)).unwrap();
        gate.send(Err(BackendError::Inference("completion failed".into())))
            .unwrap();
        assert!(matches!(
            result.blocking_recv().unwrap(),
            Err(BackendError::Inference(_))
        ));
        start.recv_timeout(Duration::from_secs(5)).unwrap();
        gate.send(Ok(())).unwrap();
        // Explicit shutdown must drain completions and drop model resources even
        // while callers retain command senders.
        commands.blocking_send(BackendCommand::Shutdown).unwrap();
        drop(worker);
        assert!(commands.is_closed());
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert_eq!(completed.load(Ordering::SeqCst), 3);
    }
}

#[cfg(all(test, any(feature = "libtorch", feature = "oxide")))]
mod dense_module_tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-dense-paths-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn modules(&self, paths: &[&str]) {
            let entries = paths
                .iter()
                .map(|path| {
                    serde_json::json!({
                        "type": "sentence_transformers.models.Dense",
                        "path": path
                    })
                })
                .collect::<Vec<_>>();
            self.write_modules(serde_json::Value::Array(entries));
        }

        fn write_modules(&self, modules: serde_json::Value) {
            fs::write(
                self.0.join("modules.json"),
                serde_json::to_vec(&modules).unwrap(),
            )
            .unwrap();
        }

        fn modules_path(&self) -> PathBuf {
            self.0.join("modules.json")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn local_override_replaces_a_single_discovered_dense_module() {
        let fixture = Fixture::new();
        fixture.modules(&["2_Dense"]);
        assert_eq!(
            local_dense_paths(&fixture.modules_path(), Some("override_Dense".into())).unwrap(),
            Some(vec!["override_Dense".into()])
        );
    }

    #[test]
    fn local_override_is_ignored_for_multiple_dense_modules() {
        let fixture = Fixture::new();
        fixture.modules(&["2_Dense", "3_Dense"]);
        assert_eq!(
            local_dense_paths(&fixture.modules_path(), Some("override_Dense".into())).unwrap(),
            Some(vec!["2_Dense".into(), "3_Dense".into()])
        );
    }

    #[test]
    fn explicit_dense_path_works_without_sentence_transformer_metadata() {
        let fixture = Fixture::new();
        assert_eq!(
            local_dense_paths(&fixture.modules_path(), Some("2_Dense".into())).unwrap(),
            Some(vec!["2_Dense".into()])
        );
        assert_eq!(
            local_dense_paths(&fixture.modules_path(), None).unwrap(),
            None
        );
    }

    #[test]
    fn malformed_dense_metadata_is_reported_for_local_overrides() {
        let fixture = Fixture::new();
        fs::write(fixture.modules_path(), b"not json").unwrap();
        assert!(local_dense_paths(&fixture.modules_path(), Some("2_Dense".into())).is_err());
    }
}

#[cfg(all(test, feature = "oxide"))]
mod oxide_dense_discovery_tests {
    use super::*;
    use std::sync::Mutex;

    struct FakeDenseSource {
        modules: ModulesFetch,
        available: Vec<String>,
        downloads: Mutex<Vec<String>>,
    }

    impl FakeDenseSource {
        fn new(modules: ModulesFetch, available: &[&str]) -> Self {
            Self {
                modules,
                available: available.iter().map(|path| (*path).into()).collect(),
                downloads: Mutex::new(Vec::new()),
            }
        }

        fn downloaded(&self) -> Vec<String> {
            self.downloads.lock().unwrap().clone()
        }
    }

    impl DenseModuleSource for FakeDenseSource {
        type ModulesFuture<'a>
            = std::future::Ready<ModulesFetch>
        where
            Self: 'a;
        type DownloadFuture<'a>
            = std::future::Ready<Result<(), String>>
        where
            Self: 'a;

        fn fetch_modules(&self) -> Self::ModulesFuture<'_> {
            std::future::ready(self.modules.clone())
        }

        fn download_module<'a>(&'a self, path: &'a str) -> Self::DownloadFuture<'a> {
            self.downloads.lock().unwrap().push(path.to_owned());
            let available = self.available.iter().any(|available| available == path);
            std::future::ready(if available {
                Ok(())
            } else {
                Err(format!("module `{path}` is missing"))
            })
        }
    }

    fn contents(modules: serde_json::Value) -> ModulesFetch {
        ModulesFetch::Contents(serde_json::to_string(&modules).unwrap())
    }

    #[test]
    fn fetch_status_classifies_only_genuine_404_as_absent() {
        assert_eq!(
            classify_modules_status(Some(404), "not found".into()),
            ModulesFetch::Absent
        );
        assert_eq!(
            classify_modules_status(Some(503), "unavailable".into()),
            ModulesFetch::Failed("HTTP 503: unavailable".into())
        );
        assert_eq!(
            classify_modules_status(None, "connection reset".into()),
            ModulesFetch::Failed("connection reset".into())
        );
    }

    #[tokio::test]
    async fn non_404_metadata_failure_fails_closed_even_with_override() {
        for override_path in [None, Some("2_Dense".to_owned())] {
            let source =
                FakeDenseSource::new(ModulesFetch::Failed("HTTP 503".into()), &["2_Dense"]);
            let error = resolve_oxide_dense_modules(&source, override_path)
                .await
                .unwrap_err();
            assert!(matches!(error, BackendError::Start(_)));
            assert!(error.to_string().contains("modules.json"), "{error}");
            assert!(source.downloaded().is_empty());
        }
    }

    #[tokio::test]
    async fn remote_404_without_override_discovers_no_dense() {
        let source = FakeDenseSource::new(ModulesFetch::Absent, &[]);
        assert_eq!(
            resolve_oxide_dense_modules(&source, None).await.unwrap(),
            None
        );
        assert!(source.downloaded().is_empty());
    }

    #[tokio::test]
    async fn remote_404_with_explicit_override_downloads_and_forwards_it() {
        let source = FakeDenseSource::new(ModulesFetch::Absent, &["2_Dense"]);
        assert_eq!(
            resolve_oxide_dense_modules(&source, Some("2_Dense".into()))
                .await
                .unwrap(),
            Some(vec!["2_Dense".into()])
        );
        assert_eq!(source.downloaded(), ["2_Dense"]);
    }

    #[tokio::test]
    async fn explicit_override_that_cannot_be_downloaded_fails_startup() {
        let source = FakeDenseSource::new(ModulesFetch::Absent, &[]);
        let error = resolve_oxide_dense_modules(&source, Some("2_Dense".into()))
            .await
            .unwrap_err();
        assert!(matches!(error, BackendError::Start(_)));
        assert!(error.to_string().contains("2_Dense"), "{error}");
        assert_eq!(source.downloaded(), ["2_Dense"]);
    }

    #[tokio::test]
    async fn zero_dense_metadata_uses_explicit_override_but_auto_empty_stays_none() {
        let modules = serde_json::json!([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}
        ]);
        let source = FakeDenseSource::new(contents(modules.clone()), &[]);
        assert_eq!(
            resolve_oxide_dense_modules(&source, None).await.unwrap(),
            None
        );
        assert!(source.downloaded().is_empty());

        let source = FakeDenseSource::new(contents(modules), &["2_Dense"]);
        assert_eq!(
            resolve_oxide_dense_modules(&source, Some("2_Dense".into()))
                .await
                .unwrap(),
            Some(vec!["2_Dense".into()])
        );
        assert_eq!(source.downloaded(), ["2_Dense"]);
    }

    #[tokio::test]
    async fn single_discovered_dense_module_can_be_overridden() {
        let modules = serde_json::json!([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
            {"type": "sentence_transformers.models.Dense", "path": "2_Dense_1024"}
        ]);
        let source = FakeDenseSource::new(contents(modules.clone()), &["2_Dense_1024"]);
        assert_eq!(
            resolve_oxide_dense_modules(&source, None).await.unwrap(),
            Some(vec!["2_Dense_1024".into()])
        );
        assert_eq!(source.downloaded(), ["2_Dense_1024"]);

        let source = FakeDenseSource::new(contents(modules), &["2_Dense", "2_Dense_1024"]);
        assert_eq!(
            resolve_oxide_dense_modules(&source, Some("2_Dense".into()))
                .await
                .unwrap(),
            Some(vec!["2_Dense".into()])
        );
        assert_eq!(source.downloaded(), ["2_Dense"]);
    }

    #[tokio::test]
    async fn multiple_dense_modules_keep_metadata_order_and_ignore_override() {
        let modules = serde_json::json!([
            {"type": "sentence_transformers.models.Dense", "path": "3_Dense"},
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
            {"type": "sentence_transformers.models.Dense", "path": "2_Dense"}
        ]);
        let source =
            FakeDenseSource::new(contents(modules), &["3_Dense", "2_Dense", "override_Dense"]);
        assert_eq!(
            // Multiple authoritative modules ignore even an unusable override.
            resolve_oxide_dense_modules(&source, Some("../ignored_Dense".into()))
                .await
                .unwrap(),
            Some(vec!["3_Dense".into(), "2_Dense".into()])
        );
        assert_eq!(source.downloaded(), ["3_Dense", "2_Dense"]);
    }

    #[tokio::test]
    async fn valid_transformer_without_path_and_three_dense_modules_load_in_order() {
        let modules = serde_json::json!([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
            {"type": "sentence_transformers.models.Dense", "path": "2_Dense"},
            {"type": "sentence_transformers.base.modules.dense.Dense", "path": "3_Dense"},
            {"type": "sentence_transformers.models.Dense", "path": "4_Dense"}
        ]);
        let source = FakeDenseSource::new(contents(modules), &["2_Dense", "3_Dense", "4_Dense"]);
        assert_eq!(
            resolve_oxide_dense_modules(&source, None).await.unwrap(),
            Some(vec!["2_Dense".into(), "3_Dense".into(), "4_Dense".into()])
        );
        assert_eq!(source.downloaded(), ["2_Dense", "3_Dense", "4_Dense"]);
    }

    #[tokio::test]
    async fn malformed_metadata_and_invalid_dense_paths_fail_before_startup() {
        for contents in [
            "not json".to_owned(),
            serde_json::json!([{"type": "sentence_transformers.models.Dense"}]).to_string(),
            serde_json::json!([{
                "type": "sentence_transformers.models.Dense",
                "path": "../escape"
            }])
            .to_string(),
            serde_json::json!([{
                "type": "sentence_transformers.models.Dense",
                "path": ""
            }])
            .to_string(),
        ] {
            let source = FakeDenseSource::new(ModulesFetch::Contents(contents), &["2_Dense"]);
            let error = resolve_oxide_dense_modules(&source, Some("2_Dense".into()))
                .await
                .unwrap_err();
            assert!(matches!(error, BackendError::Start(_)));
            assert!(source.downloaded().is_empty());
        }
    }
}
