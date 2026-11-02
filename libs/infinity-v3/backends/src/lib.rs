mod dtype;

use hf_hub::api::tokio::{ApiError, ApiRepo};
use infinity_v3_backend_core::{Backend as CoreBackend, Inference};
use std::cmp::{max, min};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{instrument, Span};

#[cfg(any(feature = "libtorch", feature = "python-backend"))]
use serde::Deserialize;

pub use crate::dtype::DType;
pub use infinity_v3_backend_core::{block_attention_config, result_memory};
pub use infinity_v3_backend_core::{
    AudioFeatures, BackendError, Batch, ClefField, DecisionInput, DecisionOutput, Embedding,
    Embeddings, ImagePatches, ModelType, MultimodalEncoding, Pool, Predictions, SharedF32,
    TokenPredictions,
};

#[cfg(feature = "python-backend")]
pub use infinity_v3_backend_python::PythonBackendConfig;

/// Parameters for a bring-your-own-Python backend.
#[cfg(not(feature = "python-backend"))]
#[derive(Clone, Debug, Default)]
pub struct PythonBackendConfig {
    pub config: Option<String>,
}

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

/// Inference runtime name retained for existing CLI configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum BackendKind {
    Libtorch,
    PythonBackend,
}

impl Default for BackendKind {
    fn default() -> Self {
        if cfg!(feature = "libtorch") {
            Self::Libtorch
        } else if cfg!(feature = "python-backend") {
            Self::PythonBackend
        } else {
            Self::Libtorch
        }
    }
}

impl BackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Libtorch => "libtorch",
            Self::PythonBackend => "python-backend",
        }
    }

    pub const fn is_compiled(self) -> bool {
        match self {
            Self::Libtorch => cfg!(feature = "libtorch"),
            Self::PythonBackend => cfg!(feature = "python-backend"),
        }
    }

    pub fn ensure_compiled(self) -> Result<(), BackendError> {
        if self.is_compiled() {
            return Ok(());
        }
        let available = [Self::Libtorch, Self::PythonBackend]
            .into_iter()
            .filter(|kind| kind.is_compiled())
            .map(Self::as_str)
            .collect::<Vec<_>>();
        Err(BackendError::Start(format!(
            "Requested backend '{}' was not compiled into this binary; compiled backends: {}.",
            self.as_str(),
            if available.is_empty() {
                "none".into()
            } else {
                available.join(", ")
            }
        )))
    }

    /// Resolve accelerator selection once before choosing dtype or replica devices.
    pub fn resolve_device(self, torch: TorchDevice) -> Result<TorchDevice, BackendError> {
        match self {
            Self::Libtorch => torch.resolved(),
            Self::PythonBackend => Ok(torch),
        }
    }

    /// Whether the selected runtime and accelerator can host GPU replicas.
    pub fn supports_replication(self, torch: TorchDevice) -> bool {
        match self {
            Self::Libtorch => torch == TorchDevice::Cuda,
            Self::PythonBackend => false,
        }
    }

    /// Query the selected runtime rather than another backend's CUDA installation.
    pub fn visible_device_count(self, torch: TorchDevice) -> Result<usize, BackendError> {
        match self {
            Self::Libtorch => torch.count(),
            Self::PythonBackend => Ok(1),
        }
    }
}

const DEFAULT_DEFERRED_COMPLETION_DEPTH: usize = 2;
const MAX_COMPLETION_QUEUE_DEPTH: usize = 256;

fn bounded_completion_depth(depth: usize) -> NonZeroUsize {
    NonZeroUsize::new(depth.min(MAX_COMPLETION_QUEUE_DEPTH)).unwrap_or(NonZeroUsize::MIN)
}

/// Resolve the deferred completion bound once before starting the worker.
/// Python admission keeps the full validated `queue_depth`; its deferred
/// completion queue is capped to avoid preallocating an oversized channel.
/// Every other backend keeps the small default matching the command channel.
fn deferred_completion_depth(
    kind: BackendKind,
    python: &PythonBackendConfig,
) -> Result<NonZeroUsize, BackendError> {
    #[cfg(feature = "python-backend")]
    if kind == BackendKind::PythonBackend {
        return Ok(bounded_completion_depth(python.queue_depth()?.get()));
    }
    let _ = (kind, python);
    Ok(bounded_completion_depth(DEFAULT_DEFERRED_COMPLETION_DEPTH))
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
            PythonBackendConfig::default(),
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
        python: PythonBackendConfig,
    ) -> Result<Self, BackendError> {
        kind.ensure_compiled()?;
        let completion_depth = deferred_completion_depth(kind, &python)?;
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
            python,
        )
        .await?;
        let radix_mlp_supported = backend.supports_radix_mlp();
        let max_batch_size = backend.max_batch_size();

        let (health_sender, health_receiver) = watch::channel(false);
        let _backend_thread = Arc::new(BackendThread::new(
            backend,
            backend_receiver,
            health_sender,
            completion_depth,
        ));

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
    python: PythonBackendConfig,
) -> Result<Box<dyn CoreBackend + Send>, BackendError> {
    if model_type == ModelType::Decision {
        if let Some(repo) = api_repo.as_ref() {
            download_decision_sidecars(repo)
                .await
                .map_err(|error| BackendError::WeightsNotFound(error.to_string()))?;
        }
    }
    if kind == BackendKind::PythonBackend {
        #[cfg(feature = "python-backend")]
        {
            if let Some(repo) = api_repo.as_ref() {
                if download_safetensors(repo.clone()).await.is_err() {
                    tracing::warn!(
                        "safetensors weights not found; falling back to pytorch_model.bin"
                    );
                    download_file(repo, "pytorch_model.bin")
                        .await
                        .map_err(|e| BackendError::WeightsNotFound(e.to_string()))?;
                }
                download_dense_modules(repo, dense_path.clone())
                    .await
                    .map_err(|e| BackendError::WeightsNotFound(e.to_string()))?;
            }
            let path = model_path.clone();
            let dtype = dtype.to_string();
            let backend = tokio::task::spawn_blocking(move || {
                infinity_v3_backend_python::PythonBackend::new(
                    &path, &dtype, model_type, device_id, python,
                )
            })
            .await
            .map_err(|error| {
                BackendError::Start(format!(
                    "Python backend initialization worker failed: {error}"
                ))
            })??;
            return Ok(Box::new(backend));
        }
        #[cfg(not(feature = "python-backend"))]
        return Err(BackendError::Start(
            "Python backend was not compiled; build with --features python".into(),
        ));
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
        } else if let Some(override_path) = dense_path {
            let modules = model_path.join("modules.json");
            if modules.exists() {
                let paths = parse_dense_paths_from_modules(&modules)
                    .await
                    .map_err(|e| BackendError::Start(e.to_string()))?;
                if paths.len() > 1 {
                    Some(paths)
                } else {
                    Some(vec![override_path])
                }
            } else {
                Some(vec![override_path])
            }
        } else {
            None
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
    fn new(depth: NonZeroUsize) -> std::io::Result<Self> {
        let capacity = depth.get().min(MAX_COMPLETION_QUEUE_DEPTH);
        let (sender, receiver) = std::sync::mpsc::sync_channel::<CompletionJob>(capacity);
        let handle = std::thread::Builder::new()
            .name("tei-completion".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    job();
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            handle: Some(handle),
        })
    }
}
impl Drop for CompletionThread {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct DeferredCompletions {
    depth: NonZeroUsize,
    thread: Option<CompletionThread>,
}

impl DeferredCompletions {
    fn new(depth: NonZeroUsize) -> Self {
        Self {
            depth,
            thread: None,
        }
    }

    fn submit(&mut self, job: CompletionJob) -> Result<(), CompletionJob> {
        if self.thread.is_none() {
            let Ok(thread) = CompletionThread::new(self.depth) else {
                return Err(job);
            };
            self.thread = Some(thread);
        }
        let Some(thread) = self.thread.as_ref() else {
            return Err(job);
        };
        let Some(sender) = thread.sender.as_ref() else {
            return Err(job);
        };
        sender.send(job).map_err(|error| error.0)
    }
}

fn deliver<T: Send + 'static>(
    result: Result<Inference<T>, BackendError>,
    sender: oneshot::Sender<Result<Submission<T>, BackendError>>,
    start: Instant,
    span: Span,
    health: &watch::Sender<bool>,
    completion: &mut DeferredCompletions,
) {
    match result {
        Ok(Inference::Deferred(complete)) => {
            let job_health = health.clone();
            let (response, receiver) = oneshot::channel();
            let job = Box::new(move || {
                let _span = span.entered();
                // Complete even after receiver cancellation: retain pins/slots until
                // the GPU finishes, and never free native work on a Tokio worker.
                let result = complete().map(|value| (value, start.elapsed()));
                let _ = job_health.send(
                    result.is_ok()
                        || result
                            .as_ref()
                            .is_err_and(|e| matches!(e, BackendError::Overloaded(_))),
                );
                let _ = response.send(result);
            });
            if let Err(job) = completion.submit(job) {
                // A closed worker must not drop native completion work: run it on
                // this backend thread so pins/slots are released before failure.
                job();
                let _ = health.send(false);
                let _ = sender.send(Err(BackendError::Unhealthy));
                return;
            }
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
        completion_depth: NonZeroUsize,
    ) -> Self {
        let handle = std::thread::spawn(move || {
            let mut completion = DeferredCompletions::new(completion_depth);
            while let Some(cmd) = backend_receiver.blocking_recv() {
                let start = Instant::now();
                match cmd {
                    BackendCommand::Shutdown => {
                        backend.shutdown();
                        break;
                    }
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

#[cfg(any(feature = "libtorch", feature = "python-backend"))]
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
    let json: serde_json::Value = serde_json::from_str(&index_file_string).map_err(|error| {
        ApiError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    })?;

    let weight_map = match json.get("weight_map") {
        Some(serde_json::Value::Object(map)) => map,
        _ => {
            return Err(ApiError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "model.safetensors.index.json is missing weight_map",
            )))
        }
    };

    let mut safetensors_filenames = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_filenames.insert(file.to_string());
        }
    }

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

#[cfg(any(feature = "libtorch", feature = "python-backend"))]
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

#[cfg(any(feature = "libtorch", feature = "python-backend"))]
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

#[cfg(any(feature = "libtorch", feature = "python-backend"))]
async fn download_file(api: &ApiRepo, file_path: &str) -> Result<PathBuf, ApiError> {
    tracing::info!("Downloading `{}`", file_path);
    api.get(file_path).await
}

#[cfg(any(feature = "libtorch", feature = "python-backend"))]
async fn parse_dense_paths_from_modules(
    modules_path: &PathBuf,
) -> Result<Vec<String>, std::io::Error> {
    let content = std::fs::read_to_string(modules_path)?;
    let modules: Vec<ModuleConfig> = serde_json::from_str(&content)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;

    Ok(modules
        .into_iter()
        .filter(|module| module.module_type == ModuleType::Dense)
        .map(|module| module.path)
        .collect::<Vec<String>>())
}

#[cfg(any(feature = "libtorch", feature = "python-backend"))]
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

#[cfg(any(feature = "libtorch", feature = "python-backend"))]
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
    fn default_completion_depth() -> NonZeroUsize {
        deferred_completion_depth(BackendKind::Libtorch, &PythonBackendConfig::default()).unwrap()
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
        let worker = BackendThread::new(
            Box::new(AdmittedBackend(pool.clone())),
            receiver,
            health,
            default_completion_depth(),
        );
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
            let worker = BackendThread::new(
                Box::new(AdmittedBackend(pool.clone())),
                receiver,
                health,
                default_completion_depth(),
            );
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
        let worker = BackendThread::new(backend, receiver, health, default_completion_depth());
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

    #[cfg(feature = "python-backend")]
    #[test]
    fn configured_python_queue_depth_admits_more_than_eleven_deferred_submissions() {
        fn wait_deferred(
            runtime: &tokio::runtime::Runtime,
            receiver: oneshot::Receiver<Result<Submission<Embeddings>, BackendError>>,
        ) -> Option<Result<Submission<Embeddings>, BackendError>> {
            runtime.block_on(async {
                match tokio::time::timeout(Duration::from_secs(5), receiver).await {
                    Ok(result) => result.ok(),
                    Err(_) => None,
                }
            })
        }

        let depth = deferred_completion_depth(
            BackendKind::PythonBackend,
            &PythonBackendConfig {
                config: Some(r#"{"queue_depth":32}"#.into()),
            },
        )
        .unwrap();
        assert_eq!(depth.get(), 32);

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
        let worker = BackendThread::new(backend, receiver, health, depth);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();

        let (response, first) = oneshot::channel();
        commands
            .blocking_send(BackendCommand::Embed(batch(), Span::none(), response))
            .unwrap();
        let mut failure = match wait_deferred(&runtime, first) {
            Some(Ok(Submission::Deferred(_))) => {
                if start.recv_timeout(Duration::from_secs(5)).is_ok() {
                    None
                } else {
                    Some("first deferred completion did not start".to_string())
                }
            }
            Some(Ok(Submission::Ready(..))) => {
                Some("expected the first submission to be deferred".to_string())
            }
            Some(Err(error)) => Some(format!("first submission failed: {error}")),
            None => Some("first deferred submission timed out".to_string()),
        };
        // The first completion is held in the worker while the configured
        // depth admits a full Python-sized queue beyond the old eleven slots.
        if failure.is_none() {
            for index in 0..depth.get() {
                let (response, submission) = oneshot::channel();
                commands
                    .blocking_send(BackendCommand::Embed(batch(), Span::none(), response))
                    .unwrap();
                match wait_deferred(&runtime, submission) {
                    Some(Ok(Submission::Deferred(_))) => {}
                    Some(Ok(Submission::Ready(..))) => {
                        failure = Some(format!("submission {index} was not deferred"));
                        break;
                    }
                    Some(Err(error)) => {
                        failure = Some(format!("submission {index} failed: {error}"));
                        break;
                    }
                    None => {
                        failure = Some(format!(
                            "submission {index} did not reach deferred completion"
                        ));
                        break;
                    }
                }
            }
        }

        let completed_before_release = completed.load(Ordering::SeqCst);
        let probes_before_release = probes.load(Ordering::SeqCst);
        // Always release and drain held completions before any assertion so a
        // shallow queue cannot leave the worker join blocked on test unwinding.
        for _ in 0..=depth.get() {
            let _ = gate.send(Ok(()));
        }
        let _ = commands.blocking_send(BackendCommand::Shutdown);
        drop(worker);
        assert!(commands.is_closed());
        if let Some(message) = failure {
            panic!("{message}");
        }
        assert_eq!(completed_before_release, 0);
        assert_eq!(probes_before_release, 0);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert_eq!(completed.load(Ordering::SeqCst), depth.get() + 1);
    }

    fn assert_bounded_completion_queue(depth: NonZeroUsize) {
        let completion = CompletionThread::new(depth).unwrap();
        let (release, wait) = std::sync::mpsc::channel();
        let (started, start) = std::sync::mpsc::channel();
        completion
            .sender
            .as_ref()
            .unwrap()
            .send(Box::new(move || {
                started.send(()).unwrap();
                let _ = wait.recv();
            }))
            .unwrap();
        start.recv_timeout(Duration::from_secs(5)).unwrap();
        for _ in 0..depth.get() {
            completion
                .sender
                .as_ref()
                .unwrap()
                .try_send(Box::new(|| {}))
                .unwrap();
        }
        assert!(matches!(
            completion
                .sender
                .as_ref()
                .unwrap()
                .try_send(Box::new(|| {})),
            Err(std::sync::mpsc::TrySendError::Full(_))
        ));
        release.send(()).unwrap();
        drop(completion);
    }

    #[test]
    fn default_completion_depth_stays_small_and_bounded() {
        let depth = default_completion_depth();
        assert_eq!(depth.get(), 2);
        assert_bounded_completion_queue(depth);
    }

    #[cfg(feature = "python-backend")]
    #[test]
    fn huge_python_queue_depth_uses_bounded_completion_capacity() {
        let python = PythonBackendConfig {
            config: Some(format!(
                r#"{{"queue_depth":{}}}"#,
                tokio::sync::Semaphore::MAX_PERMITS
            )),
        };
        let admission = python.queue_depth().unwrap();
        assert_eq!(admission.get(), tokio::sync::Semaphore::MAX_PERMITS);
        let depth = deferred_completion_depth(BackendKind::PythonBackend, &python).unwrap();
        assert_eq!(depth.get(), MAX_COMPLETION_QUEUE_DEPTH);
        assert_bounded_completion_queue(depth);
    }

    struct CompletionGuard {
        completed: Arc<AtomicUsize>,
        dropped_after_completion: Arc<AtomicUsize>,
    }

    impl Drop for CompletionGuard {
        fn drop(&mut self) {
            self.dropped_after_completion
                .store(self.completed.load(Ordering::SeqCst), Ordering::SeqCst);
        }
    }

    #[test]
    fn closed_completion_worker_runs_native_completion_before_unhealthy() {
        let completed = Arc::new(AtomicUsize::new(0));
        let dropped_after_completion = Arc::new(AtomicUsize::new(0));
        let (native_sender, native_receiver) = std::sync::mpsc::sync_channel::<CompletionJob>(1);
        drop(native_receiver);
        let mut completion = DeferredCompletions {
            depth: default_completion_depth(),
            thread: Some(CompletionThread {
                sender: Some(native_sender),
                handle: Some(std::thread::spawn(|| {})),
            }),
        };
        let (sender, response) = oneshot::channel();
        let (health, health_receiver) = watch::channel(true);
        let guard = CompletionGuard {
            completed: completed.clone(),
            dropped_after_completion: dropped_after_completion.clone(),
        };
        let completion_work = completed.clone();
        let result: Result<Inference<Embeddings>, BackendError> =
            Ok(Inference::Deferred(Box::new(move || {
                completion_work.fetch_add(1, Ordering::SeqCst);
                drop(guard);
                Ok(Embeddings::default())
            })));
        deliver(
            result,
            sender,
            Instant::now(),
            Span::none(),
            &health,
            &mut completion,
        );
        assert_eq!(completed.load(Ordering::SeqCst), 1);
        assert_eq!(dropped_after_completion.load(Ordering::SeqCst), 1);
        assert!(!*health_receiver.borrow());
        assert!(matches!(
            response.blocking_recv().unwrap(),
            Err(BackendError::Unhealthy)
        ));
    }
}

#[cfg(test)]
mod backend_selection_tests {
    use super::*;

    #[test]
    fn default_uses_compiled_backend() {
        let default = BackendKind::default();
        if cfg!(feature = "libtorch") {
            assert_eq!(default, BackendKind::Libtorch);
        } else if cfg!(feature = "python-backend") {
            assert_eq!(default, BackendKind::PythonBackend);
        } else {
            assert!(default.ensure_compiled().is_err());
        }
    }

    #[test]
    fn unavailable_backend_reports_compiled_choices() {
        for kind in [BackendKind::Libtorch, BackendKind::PythonBackend] {
            let result = kind.ensure_compiled();
            if kind.is_compiled() {
                assert!(result.is_ok());
            } else {
                let error = result.unwrap_err().to_string();
                assert!(error.contains(&format!("Requested backend '{}'", kind.as_str())));
            }
        }
    }
}
