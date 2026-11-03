/// Payload tokenization logic
use crate::TextEmbeddingsError;
use infinity_v3_backend::BackendError;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokenizers::tokenizer::Tokenizer;
pub use tokenizers::Encoding as RawEncoding;
use tokenizers::{TruncationDirection, TruncationParams, TruncationStrategy};
use tokio::sync::oneshot;
use tracing::{instrument, Span};

static MAX_CHAR_MULTIPLIER: usize = 250;

/// Maximum tokenizer request workers; BPE parallelism uses the library's own pool.
pub const MAX_TOKENIZATION_WORKERS: usize = 16;

/// Waiting tokenizer jobs, independent of the worker count.
pub const DEFAULT_TOKENIZATION_QUEUE_CAPACITY: usize = 1024;

/// Validation
#[derive(Debug, Clone)]
pub struct Tokenization {
    workers: Arc<TokenizerWorkers>,
    multimodal: Option<crate::multimodal::Qwen3VlProcessor>,
    mistral3: Option<crate::multimodal::Mistral3Processor>,
    embedding_gemma2: Option<crate::multimodal::EmbeddingGemma2Processor>,
    token_input_vocab_size: Option<usize>,
    max_input_length: usize,
    position_offset: usize,
    /// Channel to communicate with the background tokenization task
    sender: async_channel::Sender<QueuedTokenizerRequest>,
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct SimpleToken {
    pub id: u32,
    pub text: String,
    pub special: bool,
    pub start: Option<usize>,
    pub stop: Option<usize>,
}

struct TokenizerWorkerConfig {
    workers: usize,
    receiver: async_channel::Receiver<QueuedTokenizerRequest>,
    tokenizer: Tokenizer,
    accelerate_pooled_text: bool,
    max_input_length: usize,
    position_offset: usize,
    default_prompt: Option<String>,
    prompts: Option<HashMap<String, String>>,
    chat: Option<Arc<crate::chat::ChatProcessor>>,
}

enum WorkerState {
    Idle(Option<Box<TokenizerWorkerConfig>>),
    Started(Option<std::thread::JoinHandle<()>>),
    Failed,
    Shutdown,
}

struct TokenizerWorkers {
    sender: async_channel::Sender<QueuedTokenizerRequest>,
    state: std::sync::Mutex<WorkerState>,
    started: AtomicBool,
    #[cfg(test)]
    supervisor_starts: std::sync::atomic::AtomicUsize,
}

impl std::fmt::Debug for TokenizerWorkers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerWorkers").finish_non_exhaustive()
    }
}

fn workers_shutdown_error() -> TextEmbeddingsError {
    TextEmbeddingsError::Backend(BackendError::Inference(
        "Tokenization workers are shutting down; the request was not queued".into(),
    ))
}

fn workers_start_error() -> TextEmbeddingsError {
    TextEmbeddingsError::Backend(BackendError::Start(
        "Tokenization workers failed to start".into(),
    ))
}

fn workers_response_error() -> TextEmbeddingsError {
    TextEmbeddingsError::Backend(BackendError::Inference(
        "Tokenization workers stopped before responding".into(),
    ))
}

impl TokenizerWorkers {
    fn ensure_started(&self) -> Result<(), TextEmbeddingsError> {
        if self.started.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut state = self.state.lock().map_err(|_| workers_start_error())?;
        if self.started.load(Ordering::Acquire) {
            return Ok(());
        }
        let config = match &mut *state {
            WorkerState::Idle(config) => config.take(),
            WorkerState::Started(_) => {
                self.started.store(true, Ordering::Release);
                return Ok(());
            }
            WorkerState::Failed => return Err(workers_start_error()),
            WorkerState::Shutdown => return Err(workers_shutdown_error()),
        };
        let config = match config {
            Some(config) => *config,
            None => {
                *state = WorkerState::Failed;
                return Err(workers_start_error());
            }
        };
        let handle = match spawn_worker_supervisor(config) {
            Ok(handle) => handle,
            Err(error) => {
                tracing::error!(%error, "Failed to start tokenization supervisor");
                *state = WorkerState::Failed;
                return Err(workers_start_error());
            }
        };
        #[cfg(test)]
        self.supervisor_starts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        *state = WorkerState::Started(Some(handle));
        self.started.store(true, Ordering::Release);
        Ok(())
    }
}

impl Drop for TokenizerWorkers {
    fn drop(&mut self) {
        self.sender.close();
        let state = match self.state.get_mut() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let WorkerState::Started(handle) = state {
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
    }
}

impl Tokenization {
    pub async fn shutdown(&self) {
        self.sender.close();
        let handle = {
            let mut state = match self.workers.state.lock() {
                Ok(state) => state,
                Err(poisoned) => poisoned.into_inner(),
            };
            let handle = match &mut *state {
                WorkerState::Started(handle) => handle.take(),
                WorkerState::Idle(config) => {
                    config.take();
                    None
                }
                WorkerState::Failed | WorkerState::Shutdown => None,
            };
            *state = WorkerState::Shutdown;
            handle
        };
        if let Some(handle) = handle {
            let _ = tokio::task::spawn_blocking(move || handle.join()).await;
        }
    }

    pub fn new(
        workers: usize,
        tokenizer: Tokenizer,
        max_input_length: usize,
        position_offset: usize,
        default_prompt: Option<String>,
        prompts: Option<HashMap<String, String>>,
        chat: Option<crate::chat::ChatProcessor>,
    ) -> Self {
        Self::with_queue_capacity(
            workers,
            NonZeroUsize::new(DEFAULT_TOKENIZATION_QUEUE_CAPACITY).unwrap(),
            tokenizer,
            max_input_length,
            position_offset,
            default_prompt,
            prompts,
            chat,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_queue_capacity(
        workers: usize,
        queue_capacity: NonZeroUsize,
        tokenizer: Tokenizer,
        max_input_length: usize,
        position_offset: usize,
        default_prompt: Option<String>,
        prompts: Option<HashMap<String, String>>,
        chat: Option<crate::chat::ChatProcessor>,
    ) -> Self {
        Self::with_queue_capacity_and_pooled_text_acceleration(
            workers,
            queue_capacity,
            tokenizer,
            max_input_length,
            position_offset,
            default_prompt,
            prompts,
            chat,
            true,
        )
    }

    /// Build tokenization workers that start on the first queued request.
    ///
    /// This is intended for metadata-only tokenization where no pooled-text
    /// Basetenkenizer is needed. Existing constructors keep their eager
    /// startup behavior.
    #[allow(clippy::too_many_arguments)]
    pub fn with_lazy_queue_capacity(
        workers: usize,
        queue_capacity: NonZeroUsize,
        tokenizer: Tokenizer,
        max_input_length: usize,
        position_offset: usize,
        default_prompt: Option<String>,
        prompts: Option<HashMap<String, String>>,
        chat: Option<crate::chat::ChatProcessor>,
    ) -> Self {
        Self::with_queue_capacity_and_start_mode(
            workers,
            queue_capacity,
            tokenizer,
            max_input_length,
            position_offset,
            default_prompt,
            prompts,
            chat,
            false,
            true,
        )
    }

    /// Build tokenization workers with an explicit pooled-text acceleration choice.
    ///
    /// `accelerate_pooled_text` controls only the Basetenkenizer used by
    /// embedding/classifier `IdsOnly` encoding. Metadata `tokenize`/`decode`
    /// workers and multimodal processor tokenizers are unaffected.
    #[allow(clippy::too_many_arguments)]
    pub fn with_queue_capacity_and_pooled_text_acceleration(
        workers: usize,
        queue_capacity: NonZeroUsize,
        tokenizer: Tokenizer,
        max_input_length: usize,
        position_offset: usize,
        default_prompt: Option<String>,
        prompts: Option<HashMap<String, String>>,
        chat: Option<crate::chat::ChatProcessor>,
        accelerate_pooled_text: bool,
    ) -> Self {
        Self::with_queue_capacity_and_start_mode(
            workers,
            queue_capacity,
            tokenizer,
            max_input_length,
            position_offset,
            default_prompt,
            prompts,
            chat,
            accelerate_pooled_text,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_queue_capacity_and_start_mode(
        workers: usize,
        queue_capacity: NonZeroUsize,
        tokenizer: Tokenizer,
        max_input_length: usize,
        position_offset: usize,
        default_prompt: Option<String>,
        prompts: Option<HashMap<String, String>>,
        chat: Option<crate::chat::ChatProcessor>,
        accelerate_pooled_text: bool,
        lazy: bool,
    ) -> Self {
        let workers = workers.clamp(1, MAX_TOKENIZATION_WORKERS);
        let chat = chat.map(Arc::new);
        tracing::info!(
            workers,
            queue_capacity = queue_capacity.get(),
            accelerate_pooled_text,
            lazy,
            "Configuring tokenization workers"
        );

        let (sender, receiver) = async_channel::bounded(queue_capacity.get());
        let config = TokenizerWorkerConfig {
            workers,
            receiver,
            tokenizer,
            accelerate_pooled_text,
            max_input_length,
            position_offset,
            default_prompt,
            prompts,
            chat,
        };
        let state = if lazy {
            WorkerState::Idle(Some(Box::new(config)))
        } else {
            match spawn_worker_supervisor(config) {
                Ok(handle) => WorkerState::Started(Some(handle)),
                Err(error) => {
                    tracing::error!(%error, "Failed to start tokenization supervisor");
                    WorkerState::Failed
                }
            }
        };
        let started = matches!(&state, WorkerState::Started(_));
        #[cfg(test)]
        let supervisor_starts = usize::from(started);

        Self {
            workers: Arc::new(TokenizerWorkers {
                sender: sender.clone(),
                state: std::sync::Mutex::new(state),
                started: AtomicBool::new(started),
                #[cfg(test)]
                supervisor_starts: std::sync::atomic::AtomicUsize::new(supervisor_starts),
            }),
            sender,
            multimodal: None,
            mistral3: None,
            embedding_gemma2: None,
            token_input_vocab_size: None,
            max_input_length,
            position_offset,
        }
    }

    /// Use the model's embedding vocabulary, which may differ from the tokenizer's.
    pub fn with_token_input_vocab_size(mut self, vocab_size: Option<usize>) -> Self {
        self.token_input_vocab_size = vocab_size;
        self
    }

    pub fn with_mistral3(mut self, processor: crate::multimodal::Mistral3Processor) -> Self {
        self.mistral3 = Some(processor);
        self
    }

    pub fn with_multimodal(mut self, processor: crate::multimodal::Qwen3VlProcessor) -> Self {
        self.multimodal = Some(processor);
        self
    }

    #[instrument(skip_all)]
    pub async fn encode(
        &self,
        inputs: EncodingInput,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
    ) -> Result<ValidEncoding, TextEmbeddingsError> {
        self.encode_with_mode(
            inputs,
            truncate,
            truncation_direction,
            prompt_name,
            EncodeMode::Aligned,
        )
        .await
    }

    /// Sequence classification needs model inputs, not token text or source offsets.
    #[instrument(skip_all)]
    pub async fn encode_classification(
        &self,
        inputs: EncodingInput,
        truncate: bool,
        truncation_direction: TruncationDirection,
    ) -> Result<ValidEncoding, TextEmbeddingsError> {
        self.encode_with_mode(
            inputs,
            truncate,
            truncation_direction,
            None,
            EncodeMode::IdsOnly,
        )
        .await
    }

    pub fn with_embedding_gemma2(
        mut self,
        processor: crate::multimodal::EmbeddingGemma2Processor,
    ) -> Self {
        self.embedding_gemma2 = Some(processor);
        self
    }

    /// Encode embedding inputs without requesting token text or source offsets.
    #[instrument(skip_all)]
    pub async fn encode_embedding(
        &self,
        inputs: EncodingInput,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
    ) -> Result<ValidEncoding, TextEmbeddingsError> {
        if let Some(processor) = &self.embedding_gemma2 {
            if let EncodingInput::Messages(messages) = inputs {
                let prepared = processor
                    .prepare(messages, truncate, truncation_direction, prompt_name)
                    .await?;
                let length = prepared.input_ids.len();
                return Ok(ValidEncoding {
                    multimodal: Some(prepared.media),
                    input_ids: prepared.input_ids,
                    token_type_ids: vec![0; length],
                    position_ids: (0..length as u32).collect(),
                    tokens: vec![],
                    offsets: vec![],
                });
            }
        }
        if let Some(processor) = &self.mistral3 {
            let prepared = processor
                .prepare(inputs, truncate, truncation_direction, prompt_name)
                .await?;
            let length = prepared.input_ids.len();
            return Ok(ValidEncoding {
                multimodal: Some(prepared.media),
                input_ids: prepared.input_ids,
                token_type_ids: vec![0; length],
                position_ids: (0..length as u32).collect(),
                tokens: vec![],
                offsets: vec![],
            });
        }
        if let Some(processor) = &self.multimodal {
            if processor.accepts_raw_text()
                && matches!(&inputs, EncodingInput::Single(_) | EncodingInput::Ids(_))
            {
                return self
                    .encode_request(
                        inputs,
                        truncate,
                        truncation_direction,
                        prompt_name,
                        EncodeMode::IdsOnly,
                    )
                    .await;
            }
            if inputs.is_empty() {
                return Err(TextEmbeddingsError::Empty(
                    "`inputs` cannot be empty".into(),
                ));
            }
            let prepared = processor
                .prepare(inputs, truncate, truncation_direction, prompt_name)
                .await?;
            let length = prepared.input_ids.len();
            return Ok(ValidEncoding {
                multimodal: Some(prepared.media),
                input_ids: prepared.input_ids,
                token_type_ids: vec![0; length],
                position_ids: (0..length as u32).collect(),
                tokens: vec![],
                offsets: vec![],
            });
        }
        self.encode_request(
            inputs,
            truncate,
            truncation_direction,
            prompt_name,
            EncodeMode::IdsOnly,
        )
        .await
    }

    async fn encode_with_mode(
        &self,
        inputs: EncodingInput,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        mode: EncodeMode,
    ) -> Result<ValidEncoding, TextEmbeddingsError> {
        // Classification pairs and native messages retain their multimodal processing.
        if self.mistral3.is_some()
            && matches!(inputs, EncodingInput::Dual(..) | EncodingInput::Messages(_))
        {
            return self
                .encode_embedding(inputs, truncate, truncation_direction, prompt_name)
                .await;
        }
        self.encode_request(inputs, truncate, truncation_direction, prompt_name, mode)
            .await
    }

    async fn encode_request(
        &self,
        inputs: EncodingInput,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        mode: EncodeMode,
    ) -> Result<ValidEncoding, TextEmbeddingsError> {
        if let EncodingInput::Ids(mut ids) = inputs {
            if ids.is_empty() {
                return Err(TextEmbeddingsError::Empty(
                    "`inputs` cannot be empty".into(),
                ));
            }
            if prompt_name.is_some() {
                return Err(TextEmbeddingsError::Validation(
                    "`prompt_name` cannot be combined with pretokenized inputs".into(),
                ));
            }
            let vocab_size = self.token_input_vocab_size.ok_or_else(|| {
                TextEmbeddingsError::Validation(
                    "Pretokenized inputs require a model vocabulary size".into(),
                )
            })?;
            // Validate the entire submitted input, including any truncated tokens.
            if ids.iter().any(|&id| id as usize >= vocab_size) {
                return Err(TextEmbeddingsError::Validation(format!(
                    "Token IDs must be less than the model vocabulary size ({vocab_size})"
                )));
            }
            if ids.len() > self.max_input_length {
                if !truncate {
                    return Err(TextEmbeddingsError::Validation(format!(
                        "`inputs` must have at most {} tokens. Given: {}",
                        self.max_input_length,
                        ids.len()
                    )));
                }
                match truncation_direction {
                    TruncationDirection::Left => {
                        ids.drain(..ids.len() - self.max_input_length);
                    }
                    TruncationDirection::Right => ids.truncate(self.max_input_length),
                }
            }
            let length = ids.len();
            metrics::histogram!("infinity_request_input_length").record(length as f64);
            return Ok(ValidEncoding {
                multimodal: None,
                input_ids: ids,
                token_type_ids: vec![0; length],
                position_ids: (self.position_offset as u32..(length + self.position_offset) as u32)
                    .collect(),
                tokens: vec![],
                offsets: vec![],
            });
        }
        // Check if inputs is empty
        if inputs.is_empty() {
            return Err(TextEmbeddingsError::Empty(
                "`inputs` cannot be empty".to_string(),
            ));
        }

        self.workers.ensure_started()?;
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send request to the background validation task
        self.sender
            .send(QueuedTokenizerRequest::new(TokenizerRequest::Encode(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                mode,
                response_sender,
                Span::current(),
            )))
            .await
            .map_err(|_| workers_shutdown_error())?;

        // Await on response channel
        response_receiver
            .await
            .map_err(|_| workers_response_error())?
    }

    #[instrument(skip_all)]
    pub async fn tokenize(
        &self,
        inputs: EncodingInput,
        add_special_tokens: bool,
        prompt_name: Option<String>,
    ) -> Result<(Option<String>, RawEncoding), TextEmbeddingsError> {
        // Check if inputs is empty
        if inputs.is_empty() {
            return Err(TextEmbeddingsError::Empty(
                "`inputs` cannot be empty".to_string(),
            ));
        }

        self.workers.ensure_started()?;
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send request to the background validation task
        self.sender
            .send(QueuedTokenizerRequest::new(TokenizerRequest::Tokenize(
                inputs,
                add_special_tokens,
                prompt_name,
                response_sender,
                Span::current(),
            )))
            .await
            .map_err(|_| workers_shutdown_error())?;

        // Await on response channel
        response_receiver
            .await
            .map_err(|_| workers_response_error())?
    }

    #[instrument(skip_all)]
    pub async fn decode(
        &self,
        ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> Result<String, TextEmbeddingsError> {
        // Check if input_ids is empty
        if ids.is_empty() {
            return Err(TextEmbeddingsError::Empty(
                "`input_ids` cannot be empty".to_string(),
            ));
        }

        self.workers.ensure_started()?;
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send request to the background validation task
        self.sender
            .send(QueuedTokenizerRequest::new(TokenizerRequest::Decode(
                ids,
                skip_special_tokens,
                response_sender,
                Span::current(),
            )))
            .await
            .map_err(|_| workers_shutdown_error())?;

        // Await on response channel
        response_receiver
            .await
            .map_err(|_| workers_response_error())?
    }
}

fn spawn_worker_supervisor(
    config: TokenizerWorkerConfig,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("tokenization-supervisor".into())
        .spawn(move || run_tokenizer_supervisor(config))
}

#[allow(clippy::too_many_arguments)]
fn spawn_tokenizer_worker(
    tokenizer: Tokenizer,
    fast_tokenizer: crate::fast_tokenization::BackgroundFastTokenizer,
    max_input_length: usize,
    position_offset: usize,
    default_prompt: Option<String>,
    prompts: Option<HashMap<String, String>>,
    receiver: async_channel::Receiver<QueuedTokenizerRequest>,
    chat: Option<Arc<crate::chat::ChatProcessor>>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("tokenization-worker".into())
        .spawn(move || {
            tokenizer_worker(
                tokenizer,
                fast_tokenizer,
                max_input_length,
                position_offset,
                default_prompt,
                prompts,
                receiver,
                chat,
            )
        })
}

fn run_tokenizer_supervisor(config: TokenizerWorkerConfig) {
    let accelerate =
        config.accelerate_pooled_text && crate::fast_tokenization::is_bpe(&config.tokenizer);
    // Keep one clone for the builder and remaining workers. Worker 1 consumes
    // the original so it can serve HF requests without waiting for Baseten.
    let seed = if !config.receiver.is_closed() && (config.workers > 1 || accelerate) {
        Some(Arc::new(config.tokenizer.clone()))
    } else {
        None
    };
    let fast_tokenizer = if accelerate {
        match seed.as_ref() {
            Some(seed) => crate::fast_tokenization::BackgroundFastTokenizer::new(Arc::clone(seed)),
            None => crate::fast_tokenization::BackgroundFastTokenizer::disabled(),
        }
    } else {
        crate::fast_tokenization::BackgroundFastTokenizer::disabled()
    };
    run_worker_pool(config, seed, fast_tokenizer);
}

fn run_worker_pool(
    config: TokenizerWorkerConfig,
    mut seed: Option<Arc<Tokenizer>>,
    fast_tokenizer: crate::fast_tokenization::BackgroundFastTokenizer,
) {
    let TokenizerWorkerConfig {
        workers,
        receiver,
        tokenizer,
        max_input_length,
        position_offset,
        default_prompt,
        prompts,
        chat,
        ..
    } = config;
    // A single worker consumes the original tokenizer. Drop any unused seed here
    // so it cannot outlive a completed background builder.
    if workers <= 1 {
        seed = None;
    }
    let mut handles = Vec::with_capacity(workers);
    // The first worker always starts so a request queued before shutdown can
    // drain. Once the queue is closed, skip cloning and spawning the rest.
    let worker = spawn_tokenizer_worker(
        tokenizer,
        fast_tokenizer.clone(),
        max_input_length,
        position_offset,
        default_prompt.clone(),
        prompts.clone(),
        receiver.clone(),
        chat.clone(),
    );
    match worker {
        Ok(handle) => handles.push(handle),
        Err(error) => tracing::error!(%error, "Failed to start tokenization worker"),
    }
    if workers > 1 && seed.is_some() {
        for index in 1..workers {
            if receiver.is_closed() {
                break;
            }
            let tokenizer_clone = if index + 1 == workers {
                match seed.take() {
                    Some(seed) => match Arc::try_unwrap(seed) {
                        Ok(tokenizer) => tokenizer,
                        Err(seed) => (*seed).clone(),
                    },
                    None => break,
                }
            } else {
                match seed.as_ref() {
                    Some(seed) => (**seed).clone(),
                    None => break,
                }
            };
            let worker = spawn_tokenizer_worker(
                tokenizer_clone,
                fast_tokenizer.clone(),
                max_input_length,
                position_offset,
                default_prompt.clone(),
                prompts.clone(),
                receiver.clone(),
                chat.clone(),
            );
            match worker {
                Ok(handle) => handles.push(handle),
                Err(error) => tracing::error!(%error, "Failed to start tokenization worker"),
            }
        }
    }
    for handle in handles {
        let _ = handle.join();
    }
}

/// Start tokenization workers
#[allow(clippy::too_many_arguments)]
fn tokenizer_worker(
    mut tokenizer: Tokenizer,
    fast_tokenizer: crate::fast_tokenization::BackgroundFastTokenizer,
    max_input_length: usize,
    position_offset: usize,
    default_prompt: Option<String>,
    prompts: Option<HashMap<String, String>>,
    receiver: async_channel::Receiver<QueuedTokenizerRequest>,
    chat: Option<Arc<crate::chat::ChatProcessor>>,
) {
    // Publication is permanent. Cache the borrowed accelerator per worker so
    // steady requests no longer need to check the shared initialization state.
    let mut ready_fast_tokenizer = None;
    // Loop over requests
    while let Ok(queued) = receiver.recv_blocking() {
        // Keep admission through processing and response delivery, even when
        // cancellation has closed the receiver on the original HTTP task.
        let QueuedTokenizerRequest {
            request,
            _input_memory,
        } = queued;
        match request {
            TokenizerRequest::Encode(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                mode,
                response_tx,
                parent_span,
            ) => {
                parent_span.in_scope(|| {
                    if !response_tx.is_closed() {
                        let fast = if matches!(mode, EncodeMode::IdsOnly) {
                            if ready_fast_tokenizer.is_none() {
                                ready_fast_tokenizer = fast_tokenizer.get();
                            }
                            ready_fast_tokenizer
                        } else {
                            None
                        };
                        let default_prompt_clone = match prompt_name {
                            None => default_prompt.clone(),
                            Some(_) => None,
                        };

                        // It's possible that the user dropped its request resulting in a send error.
                        // We just discard the error
                        let _ = response_tx.send(encode_input(
                            inputs,
                            truncate,
                            truncation_direction,
                            max_input_length,
                            position_offset,
                            default_prompt_clone,
                            prompt_name,
                            prompts.as_ref(),
                            &mut tokenizer,
                            fast,
                            chat.as_deref(),
                            mode,
                        ));
                    }
                })
            }
            TokenizerRequest::Tokenize(
                inputs,
                add_special_tokens,
                prompt_name,
                response_tx,
                parent_span,
            ) => {
                parent_span.in_scope(|| {
                    if !response_tx.is_closed() {
                        let default_prompt_clone = match prompt_name {
                            None => default_prompt.clone(),
                            Some(_) => None,
                        };

                        // It's possible that the user dropped its request resulting in a send error.
                        // We just discard the error
                        let _ = response_tx.send(tokenize_input(
                            inputs,
                            add_special_tokens,
                            max_input_length,
                            None,
                            default_prompt_clone,
                            prompt_name,
                            prompts.as_ref(),
                            &mut tokenizer,
                            None,
                            chat.as_deref(),
                        ));
                    }
                })
            }
            TokenizerRequest::Decode(ids, skip_special_tokens, response_tx, parent_span) => {
                parent_span.in_scope(|| {
                    if !response_tx.is_closed() {
                        // It's possible that the user dropped its request resulting in a send error.
                        // We just discard the error
                        let _ =
                            response_tx.send(decode_ids(ids, skip_special_tokens, &mut tokenizer));
                    }
                })
            }
        }
    }
}

fn decode_ids(
    ids: Vec<u32>,
    skip_special_tokens: bool,
    tokenizer: &mut Tokenizer,
) -> Result<String, TextEmbeddingsError> {
    Ok(tokenizer
        .with_truncation(None)?
        .decode(&ids, skip_special_tokens)?)
}

fn prepare_pre_prompt(
    default_prompt: Option<String>,
    prompt_name: Option<String>,
    prompts: Option<&HashMap<String, String>>,
) -> Result<Option<String>, TextEmbeddingsError> {
    let pre_prompt = if let Some(prompt_name) = prompt_name.as_ref() {
        match prompts {
            None => {
                return Err(TextEmbeddingsError::Validation(format!("`default-prompt-name` is set to `{prompt_name}` but no prompts were found in the Sentence Transformers configuration")));
            }
            Some(prompts) if !prompts.contains_key(prompt_name) => {
                return Err(TextEmbeddingsError::Validation(format!("`default-prompt-name` is set to `{prompt_name}` but it was not found in the Sentence Transformers prompts. Available prompts: {:?}", prompts.keys())));
            }
            Some(prompts) => prompts.get(prompt_name).cloned(),
        }
    } else {
        default_prompt
    };
    Ok(pre_prompt)
}

#[allow(clippy::too_many_arguments)]
fn tokenize_input(
    mut inputs: EncodingInput,
    add_special_tokens: bool,
    max_input_length: usize,
    truncate_params: Option<TruncationParams>,
    default_prompt: Option<String>,
    prompt_name: Option<String>,
    prompts: Option<&HashMap<String, String>>,
    tokenizer: &mut Tokenizer,
    fast_tokenizer: Option<&crate::fast_tokenization::FastTokenizer>,
    chat: Option<&crate::chat::ChatProcessor>,
) -> Result<(Option<String>, RawEncoding), TextEmbeddingsError> {
    let conversation = matches!(&inputs, EncodingInput::Messages(_));
    let add_special_tokens = if conversation {
        false
    } else {
        add_special_tokens
    };
    let mut input_chars = inputs.count_chars();
    if let EncodingInput::Messages(messages) = inputs {
        let processor = chat.ok_or_else(|| TextEmbeddingsError::Tokenizer(
            "The loaded model has no conversation processor; provide a string or a list of strings".into()
        ))?;
        if prompt_name.is_some() {
            return Err(TextEmbeddingsError::Tokenizer(
                "`prompt_name` cannot be combined with native messages".into(),
            ));
        }
        // Bound work before rendering; do not truncate structured message content.
        if input_chars > max_input_length * MAX_CHAR_MULTIPLIER {
            return Err(TextEmbeddingsError::Validation(
                "Conversation exceeds the input character limit".into(),
            ));
        }
        inputs = EncodingInput::Single(
            processor
                .render(messages)
                .map_err(|e| TextEmbeddingsError::Tokenizer(e.into()))?,
        );
        input_chars = inputs.count_chars();
    }
    // The native template owns instructions and special tokens for conversations.
    let default_prompt = if conversation { None } else { default_prompt };

    let pre_prompt = prepare_pre_prompt(default_prompt, prompt_name, prompts)?;
    let limit = max_input_length * MAX_CHAR_MULTIPLIER;
    if input_chars > limit {
        if conversation || truncate_params.is_none() {
            return Err(TextEmbeddingsError::Validation(format!(
                "`inputs` must have less than {limit} characters. Given: {input_chars}"
            )));
        }
        inputs.apply_limit(limit);
    }

    let encoding = match inputs {
        EncodingInput::Messages(_) => unreachable!("messages rendered before text tokenization"),
        // encode input
        EncodingInput::Single(s) => {
            let s = if let Some(mut pre_prompt) = pre_prompt {
                pre_prompt.push_str(&s);
                pre_prompt
            } else {
                s
            };

            tokenizer.with_truncation(truncate_params)?;
            let encoding = match fast_tokenizer {
                Some(fast) => {
                    match crate::fast_tokenization::encode(fast, tokenizer, &s, add_special_tokens)
                    {
                        Ok(encoding) => encoding,
                        Err(err) => {
                            tracing::debug!(
                                "Fast BPE encoding failed; using Hugging Face tokenizer: {err}"
                            );
                            tokenizer.encode::<&str>(&s, add_special_tokens)?
                        }
                    }
                }
                None => tokenizer.encode::<&str>(&s, add_special_tokens)?,
            };

            (Some(s), encoding)
        }
        EncodingInput::Dual(s1, s2) => {
            if pre_prompt.is_some() {
                return Err(TextEmbeddingsError::Validation(
                    "`prompt_name` cannot be set with dual inputs".to_string(),
                ));
            }

            (
                None,
                tokenizer
                    .with_truncation(truncate_params)?
                    .encode::<(String, String)>((s1, s2), add_special_tokens)?,
            )
        }
        // input is encoded -> convert to tokenizers Encoding
        EncodingInput::Ids(ids) => {
            if let Some(mut pre_prompt) = pre_prompt {
                let text = tokenizer.decode(&ids, true)?;
                pre_prompt.push_str(&text);

                let encoding = tokenizer
                    .with_truncation(truncate_params)?
                    .encode::<&str>(&pre_prompt, true)?;

                (Some(pre_prompt), encoding)
            } else {
                let text = tokenizer.decode(&ids, false)?;

                let encoding = tokenizer
                    .with_truncation(truncate_params)?
                    .encode::<&str>(&text, false)?;

                (Some(text), encoding)
            }
        }
    };
    Ok(encoding)
}

/// Get input length and optionally truncate it
#[allow(clippy::too_many_arguments)]
fn encode_input(
    inputs: EncodingInput,
    truncate: bool,
    truncation_direction: TruncationDirection,
    max_input_length: usize,
    position_offset: usize,
    default_prompt: Option<String>,
    prompt_name: Option<String>,
    prompts: Option<&HashMap<String, String>>,
    tokenizer: &mut Tokenizer,
    fast_tokenizer: Option<&crate::fast_tokenization::FastTokenizer>,
    chat: Option<&crate::chat::ChatProcessor>,
    mode: EncodeMode,
) -> Result<ValidEncoding, TextEmbeddingsError> {
    // Default truncation params
    let truncate_params = truncate.then_some(TruncationParams {
        direction: truncation_direction,
        max_length: max_input_length,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
    });

    let (_, encoding) = tokenize_input(
        inputs,
        true,
        max_input_length,
        truncate_params,
        default_prompt,
        prompt_name,
        prompts,
        tokenizer,
        fast_tokenizer,
        chat,
    )?;
    let seq_len = encoding.len();

    if seq_len > max_input_length {
        return Err(TextEmbeddingsError::Validation(format!(
            "`inputs` must have less than {max_input_length} tokens. Given: {seq_len}"
        )));
    }
    let histogram = metrics::histogram!("infinity_request_input_length");
    histogram.record(seq_len as f64);
    Ok(ValidEncoding {
        multimodal: None,
        input_ids: encoding.get_ids().to_vec(),
        token_type_ids: encoding.get_type_ids().to_vec(),
        position_ids: (position_offset as u32..(seq_len + position_offset) as u32)
            .collect::<Vec<_>>(),
        tokens: if matches!(mode, EncodeMode::Aligned) {
            encoding.get_tokens().to_vec()
        } else {
            Vec::new()
        },
        offsets: if matches!(mode, EncodeMode::Aligned) {
            encoding.get_offsets().to_vec()
        } else {
            Vec::new()
        },
    })
}

#[derive(Debug)]
pub struct ValidEncoding {
    pub multimodal: Option<Arc<infinity_v3_backend::MultimodalEncoding>>,
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub enum EncodingInput {
    Single(String),
    Dual(String, String),
    Ids(Vec<u32>),
    Messages(Vec<crate::input::Message>),
}

impl EncodingInput {
    fn is_empty(&self) -> bool {
        match self {
            EncodingInput::Single(s) => s.is_empty(),
            EncodingInput::Dual(s1, s2) => s1.is_empty() && s2.is_empty(),
            EncodingInput::Ids(v) => v.is_empty(),
            EncodingInput::Messages(messages) => messages.is_empty(),
        }
    }

    pub fn count_chars(&self) -> usize {
        match self {
            EncodingInput::Single(s) => s.chars().count(),
            EncodingInput::Dual(s1, s2) => s1.chars().count() + s2.chars().count(),
            // Original characters are unavailable for pretokenized inputs.
            EncodingInput::Ids(_) => 0,
            EncodingInput::Messages(messages) => messages
                .iter()
                .map(|message| {
                    use crate::input::{ContentPart, MessageContent};
                    match &message.content {
                        MessageContent::Text(text) => text.chars().count(),
                        MessageContent::Parts(parts) => parts
                            .iter()
                            .map(|part| match part {
                                ContentPart::Text { text } => text.chars().count(),
                                _ => 0,
                            })
                            .sum(),
                    }
                })
                .sum(),
        }
    }

    fn apply_limit(&mut self, limit: usize) {
        let truncate_string = |s: &mut String, limit: usize| {
            if s.is_char_boundary(limit) {
                s.truncate(limit)
            }
        };

        match self {
            EncodingInput::Single(s) => {
                truncate_string(s, limit);
            }
            EncodingInput::Dual(s1, s2) => {
                truncate_string(s1, limit / 2);
                truncate_string(s2, limit / 2);
            }
            // Conversation truncation belongs to the model processor; never cut media spans.
            EncodingInput::Ids(_) | EncodingInput::Messages(_) => {}
        }
    }
}

impl From<String> for EncodingInput {
    fn from(value: String) -> Self {
        Self::Single(value)
    }
}

impl From<(String, String)> for EncodingInput {
    fn from(value: (String, String)) -> Self {
        Self::Dual(value.0, value.1)
    }
}

#[derive(Clone, Copy)]
enum EncodeMode {
    IdsOnly,
    Aligned,
}

struct QueuedTokenizerRequest {
    request: TokenizerRequest,
    _input_memory: Option<crate::input_memory::InputMemoryOwner>,
}
impl QueuedTokenizerRequest {
    fn new(request: TokenizerRequest) -> Self {
        Self {
            request,
            _input_memory: crate::input_memory::current(),
        }
    }
}

enum TokenizerRequest {
    Encode(
        EncodingInput,
        bool,
        TruncationDirection,
        Option<String>,
        EncodeMode,
        oneshot::Sender<Result<ValidEncoding, TextEmbeddingsError>>,
        Span,
    ),
    Tokenize(
        EncodingInput,
        bool,
        Option<String>,
        oneshot::Sender<Result<(Option<String>, RawEncoding), TextEmbeddingsError>>,
        Span,
    ),
    Decode(
        Vec<u32>,
        bool,
        oneshot::Sender<Result<String, TextEmbeddingsError>>,
        Span,
    ),
}

pub fn into_tokens(encoding: tokenizers::Encoding, input: &str) -> Vec<SimpleToken> {
    encoding
        .get_ids()
        .iter()
        .zip(encoding.get_offsets())
        .zip(encoding.get_special_tokens_mask())
        .zip(encoding.get_tokens())
        .map(|(((&id, &(start, stop)), special), token)| {
            let special = *special == 1;
            match special {
                true => SimpleToken {
                    id,
                    text: token.clone(),
                    special,
                    start: None,
                    stop: None,
                },
                false => {
                    let bytes = input.as_bytes().get(start..).unwrap_or_default();
                    let bytes = &bytes[..(stop - start).min(bytes.len())];
                    let text = String::from_utf8_lossy(bytes).into_owned();
                    SimpleToken {
                        id,
                        text,
                        special,
                        start: Some(start),
                        stop: Some(stop),
                    }
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::sync::ApiBuilder;

    #[test]
    fn source_tokens_preserve_byte_offsets_and_special_text() {
        let offsets = vec![(0, 0), (0, 5), (6, 12), (6, 7), (9, 99), (99, 100)];
        let encoding = tokenizers::Encoding::new(
            vec![0, 1, 2, 3, 4, 5],
            vec![0; 6],
            vec!["<s>".into(); 6],
            vec![None; 6],
            offsets.clone(),
            vec![1, 0, 0, 0, 0, 0],
            vec![1; 6],
            vec![],
            Default::default(),
        );
        let tokens = into_tokens(encoding, "hello 東京");
        let expected = ["<s>", "hello", "東京", "�", "京", ""];
        for (index, token) in tokens.iter().enumerate() {
            assert_eq!(token.id, index as u32);
            assert_eq!(token.text, expected[index]);
            assert_eq!(token.special, index == 0);
            assert_eq!(token.start, (index != 0).then_some(offsets[index].0));
            assert_eq!(token.stop, (index != 0).then_some(offsets[index].1));
        }
        assert_eq!(tokens.len(), expected.len());
    }

    #[test]
    fn tokenizer() {
        let api = ApiBuilder::from_env().build().unwrap();
        let filename = api
            .model("BAAI/bge-m3".to_string())
            .get("tokenizer.json")
            .unwrap();
        let string = "这是一个文本向量化的测试句子";
        let tokenizer = Tokenizer::from_file(filename).unwrap();

        let encoded = tokenizer.encode(string, true).unwrap();
        assert_eq!(
            encoded.get_offsets(),
            vec![
                (0, 0),
                (0, 3),
                (0, 12),
                (12, 18),
                (18, 21),
                (21, 24),
                (24, 30),
                (30, 36),
                (36, 39),
                (39, 42),
                (0, 0)
            ]
        );

        let tokens = into_tokens(encoded, string);
        assert_eq!(
            tokens,
            vec![
                SimpleToken {
                    id: 0,
                    text: "<s>".to_string(),
                    special: true,
                    start: None,
                    stop: None
                },
                SimpleToken {
                    id: 6,
                    text: "这".to_string(),
                    special: false,
                    start: Some(0),
                    stop: Some(3)
                },
                SimpleToken {
                    id: 100013,
                    text: "这是一个".to_string(),
                    special: false,
                    start: Some(0),
                    stop: Some(12)
                },
                SimpleToken {
                    id: 189061,
                    text: "文本".to_string(),
                    special: false,
                    start: Some(12),
                    stop: Some(18)
                },
                SimpleToken {
                    id: 2110,
                    text: "向".to_string(),
                    special: false,
                    start: Some(18),
                    stop: Some(21)
                },
                SimpleToken {
                    id: 3272,
                    text: "量".to_string(),
                    special: false,
                    start: Some(21),
                    stop: Some(24)
                },
                SimpleToken {
                    id: 41904,
                    text: "化的".to_string(),
                    special: false,
                    start: Some(24),
                    stop: Some(30)
                },
                SimpleToken {
                    id: 49125,
                    text: "测试".to_string(),
                    special: false,
                    start: Some(30),
                    stop: Some(36)
                },
                SimpleToken {
                    id: 27683,
                    text: "句".to_string(),
                    special: false,
                    start: Some(36),
                    stop: Some(39)
                },
                SimpleToken {
                    id: 1344,
                    text: "子".to_string(),
                    special: false,
                    start: Some(39),
                    stop: Some(42)
                },
                SimpleToken {
                    id: 2,
                    text: "</s>".to_string(),
                    special: true,
                    start: None,
                    stop: None
                }
            ]
        );
    }
}

#[cfg(test)]
mod fast_embedding_tests {
    use super::*;

    #[tokio::test]
    async fn cancelled_tokenizer_job_keeps_input_admission_until_discarded() {
        let (sender, receiver) = async_channel::bounded(1);
        let tokenization = Tokenization {
            multimodal: None,
            mistral3: None,
            embedding_gemma2: None,
            token_input_vocab_size: None,
            max_input_length: 16,
            position_offset: 0,
            workers: Arc::new(TokenizerWorkers {
                sender: sender.clone(),
                state: std::sync::Mutex::new(WorkerState::Started(None)),
                started: AtomicBool::new(true),
                #[cfg(test)]
                supervisor_starts: std::sync::atomic::AtomicUsize::new(0),
            }),
            sender,
        };
        let budget = Arc::new(tokio::sync::Semaphore::new(1));
        for _ in 0..2 {
            let owner = Arc::new(budget.clone().try_acquire_owned().unwrap());
            let tokenizer = tokenization.clone();
            let request = tokio::spawn(crate::input_memory::with_input_memory(owner, async move {
                tokenizer.decode(vec![1, 2, 3], true).await
            }));
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while receiver.is_empty() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            request.abort();
            assert!(request.await.unwrap_err().is_cancelled());
            assert_eq!(budget.available_permits(), 0);
            let queued = receiver.recv().await.unwrap();
            // A paused worker can hold the job outside the channel too.
            assert_eq!(budget.available_permits(), 0);
            drop(queued);
            assert_eq!(budget.available_permits(), 1);
        }
    }

    #[test]
    fn pretokenized_embeddings_preserve_the_final_encoding() {
        let mut hf = crate::fast_tokenization::test_tokenizer();
        hf.add_special_tokens(&[tokenizers::AddedToken::from("<s>", true)]);
        hf.with_post_processor(Some(
            tokenizers::processors::template::TemplateProcessing::builder()
                .try_single("<s> $A")
                .unwrap()
                .special_tokens(vec![("<s>", 256)])
                .build()
                .unwrap(),
        ));
        let ids = hf.encode("hello", true).unwrap().get_ids().to_vec();
        let tokenizer = Tokenization::new(1, hf, 32, 2, None, None, None)
            .with_token_input_vocab_size(Some(258));
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let text = tokenizer
                .encode_embedding(
                    EncodingInput::Single("hello".into()),
                    false,
                    TruncationDirection::Right,
                    None,
                )
                .await
                .unwrap();
            let tokens = tokenizer
                .encode_embedding(
                    EncodingInput::Ids(ids.clone()),
                    false,
                    TruncationDirection::Right,
                    None,
                )
                .await
                .unwrap();
            assert_eq!(tokens.input_ids, ids);
            assert_eq!(tokens.input_ids, text.input_ids);
            assert_eq!(tokens.token_type_ids, text.token_type_ids);
            assert_eq!(tokens.position_ids, text.position_ids);
            assert!(tokens.tokens.is_empty() && tokens.offsets.is_empty());
            // ID 257 is in the embedding table but absent from the tokenizer.
            let tokens = tokenizer
                .encode_embedding(
                    EncodingInput::Ids(vec![257, 0, 256]),
                    false,
                    TruncationDirection::Right,
                    None,
                )
                .await
                .unwrap();
            assert_eq!(tokens.input_ids, vec![257, 0, 256]);
            assert_eq!(EncodingInput::Ids(ids).count_chars(), 0);
        });
    }

    #[test]
    fn pretokenized_embeddings_validate_and_truncate_without_prompting() {
        let tokenizer = Tokenization::new(
            1,
            crate::fast_tokenization::test_tokenizer(),
            3,
            2,
            Some("default prompt must not be applied".into()),
            None,
            None,
        )
        .with_token_input_vocab_size(Some(258));
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            for (direction, expected) in [
                (TruncationDirection::Left, vec![1, 2, 3]),
                (TruncationDirection::Right, vec![0, 1, 2]),
            ] {
                let encoding = tokenizer
                    .encode_embedding(EncodingInput::Ids(vec![0, 1, 2, 3]), true, direction, None)
                    .await
                    .unwrap();
                assert_eq!(encoding.input_ids, expected);
                assert_eq!(encoding.position_ids, vec![2, 3, 4]);
                assert_eq!(encoding.token_type_ids, vec![0; 3]);
            }
            for (ids, truncate, prompt) in [
                (vec![], false, None),
                (vec![0, 1, 2, 3], false, None),
                (vec![258, 0, 1, 2], true, None),
                (vec![u32::MAX], false, None),
                (vec![1], false, Some("query".into())),
            ] {
                assert!(tokenizer
                    .encode_embedding(
                        EncodingInput::Ids(ids),
                        truncate,
                        TruncationDirection::Left,
                        prompt,
                    )
                    .await
                    .is_err());
            }
            let unknown_vocab = Tokenization::new(
                1,
                crate::fast_tokenization::test_tokenizer(),
                3,
                0,
                None,
                None,
                None,
            );
            assert!(unknown_vocab
                .encode_embedding(
                    EncodingInput::Ids(vec![1]),
                    false,
                    TruncationDirection::Right,
                    None,
                )
                .await
                .is_err());
        });
    }

    #[test]
    fn text_processor_rejects_conversations_before_prompting_or_truncation() {
        use crate::input::{ContentPart, ImageSource, Message, MessageContent, MessageRole};
        let mut tokenizer = crate::fast_tokenization::test_tokenizer();
        for content in [
            MessageContent::Text("A conversation".into()),
            MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageSource {
                    url: "https://example.com/image?secret=hidden".into(),
                    detail: None,
                },
            }]),
        ] {
            let error = tokenize_input(
                EncodingInput::Messages(vec![Message {
                    role: MessageRole::User,
                    content,
                }]),
                true,
                1,
                Some(TruncationParams::default()),
                None,
                Some("missing".into()),
                None,
                &mut tokenizer,
                None,
                None,
            )
            .unwrap_err();
            assert!(matches!(error, TextEmbeddingsError::Tokenizer(_)));
            assert!(error.to_string().contains("no conversation processor"));
            assert!(!error.to_string().contains("secret"));
        }
    }

    #[tokio::test]
    async fn hf_embedding_fallback_keeps_ids_without_copying_alignment() {
        let model = tokenizers::models::wordpiece::WordPiece::builder()
            .vocab(
                [
                    ("[UNK]".into(), 0),
                    ("hello".into(), 1),
                    ("world".into(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut hf = Tokenizer::new(model);
        hf.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        let tokenization = Tokenization::new(1, hf, 3, 2, None, None, None);
        for direction in [TruncationDirection::Left, TruncationDirection::Right] {
            let text = "hello world hello world";
            let aligned = tokenization
                .encode(text.to_owned().into(), true, direction, None)
                .await
                .unwrap();
            let embedding = tokenization
                .encode_embedding(text.to_owned().into(), true, direction, None)
                .await
                .unwrap();
            assert_eq!(embedding.input_ids, aligned.input_ids);
            assert_eq!(embedding.token_type_ids, aligned.token_type_ids);
            assert_eq!(embedding.position_ids, aligned.position_ids);
            assert_eq!(embedding.input_ids.len(), 3);
            assert!(!aligned.tokens.is_empty());
            assert!(!aligned.offsets.is_empty());
            assert!(embedding.tokens.is_empty());
            assert!(embedding.offsets.is_empty());
        }
    }

    #[test]
    fn sequence_classification_preserves_ids_and_types_without_alignment() {
        let mut hf = crate::fast_tokenization::test_tokenizer();
        hf.with_post_processor(Some(
            tokenizers::processors::template::TemplateProcessing::builder()
                .try_single("$A:2")
                .unwrap()
                .try_pair("$A:2 $B:3")
                .unwrap()
                .build()
                .unwrap(),
        ));
        let tokenizer = Tokenization::new(2, hf, 32, 2, None, None, None);
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            for direction in [TruncationDirection::Left, TruncationDirection::Right] {
                for pair in [false, true] {
                    let input = || {
                        if pair {
                            EncodingInput::Dual("query 東京".into(), "a longer document 👩🏽‍💻".into())
                        } else {
                            EncodingInput::Single("hello 東京 👩🏽‍💻 longer text".into())
                        }
                    };
                    let aligned = tokenizer
                        .encode(input(), true, direction, None)
                        .await
                        .unwrap();
                    let classification = tokenizer
                        .encode_classification(input(), true, direction)
                        .await
                        .unwrap();
                    assert_eq!(classification.input_ids, aligned.input_ids);
                    assert_eq!(classification.token_type_ids, aligned.token_type_ids);
                    assert_eq!(classification.position_ids, aligned.position_ids);
                    assert!(classification.tokens.is_empty() && classification.offsets.is_empty());
                    assert!(!aligned.tokens.is_empty() && !aligned.offsets.is_empty());
                }
            }
        });
    }

    #[test]
    #[ignore = "requires TOKENIZATION_CHECKPOINT and TOKENIZATION_INPUTS fixtures"]
    fn sequence_classification_checkpoint_id_parity() {
        let root = std::path::PathBuf::from(std::env::var("TOKENIZATION_CHECKPOINT").unwrap());
        let hf = Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
        let mut inputs: Vec<String> = serde_json::from_slice(
            &std::fs::read(std::env::var("TOKENIZATION_INPUTS").unwrap()).unwrap(),
        )
        .unwrap();
        inputs.extend([
            "你好 Café 👩🏽‍💻\n<|im_start|>user\n<|im_end|>".into(),
            "Naïve\u{a0}café e\u{301}\n  tabs\tcode".into(),
        ]);
        let tokenizer = Tokenization::new(4, hf.clone(), 8192, 0, None, None, None);
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            for text in inputs {
                let expected = hf.encode(text.as_str(), true).unwrap();
                let actual = tokenizer
                    .encode_classification(text.into(), false, TruncationDirection::Right)
                    .await
                    .unwrap();
                assert_eq!(actual.input_ids, expected.get_ids());
                assert_eq!(actual.token_type_ids, expected.get_type_ids());
            }
        });
    }

    #[test]
    fn workers_preserve_prompts_truncation_and_metadata_endpoints() {
        let mut hf = crate::fast_tokenization::test_tokenizer();
        hf.add_special_tokens(&[
            tokenizers::AddedToken::from("<s>", true),
            tokenizers::AddedToken::from("</s>", true),
        ]);
        hf.with_post_processor(Some(
            tokenizers::processors::template::TemplateProcessing::builder()
                .try_single("<s>:0 $A:0 </s>:0")
                .unwrap()
                .try_pair("<s>:0 $A:0 </s>:0 $B:1 </s>:1")
                .unwrap()
                .special_tokens(vec![("<s>", 256), ("</s>", 257)])
                .build()
                .unwrap(),
        ));
        let tokenizer = Tokenization::new(
            2,
            hf,
            32,
            2,
            Some("default: ".into()),
            Some(HashMap::from([("query".into(), "query: ".into())])),
            None,
        );
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            for direction in [TruncationDirection::Left, TruncationDirection::Right] {
                for prompt in [None, Some("query".to_owned())] {
                    let text = "hello 東京 👩🏽‍💻 <s> longer text";
                    let actual = tokenizer
                        .encode_embedding(text.to_string().into(), true, direction, prompt.clone())
                        .await
                        .unwrap();
                    let expected = tokenizer
                        .encode(text.to_string().into(), true, direction, prompt.clone())
                        .await
                        .unwrap();
                    assert_eq!(actual.input_ids, expected.input_ids);
                    assert_eq!(actual.token_type_ids, expected.token_type_ids);
                    assert_eq!(actual.position_ids, expected.position_ids);
                    assert!(!expected.offsets.is_empty());
                    let decoded = tokenizer
                        .decode(expected.input_ids.clone(), false)
                        .await
                        .unwrap();
                    assert!(!decoded.is_empty());
                }
            }
            // The metadata endpoint must keep actual strings and source offsets.
            let (_, tokens) = tokenizer
                .tokenize(EncodingInput::Single("hi".into()), true, None)
                .await
                .unwrap();
            assert!(tokens.get_offsets().iter().any(|&(start, end)| end > start));
            assert!(tokens.get_tokens().iter().all(|token| !token.is_empty()));
            assert!(tokenizer
                .encode_embedding(
                    EncodingInput::Single("hello".into()),
                    true,
                    TruncationDirection::Right,
                    Some("missing".into())
                )
                .await
                .is_err());
            assert!(tokenizer
                .encode_embedding(
                    EncodingInput::Single("x".repeat(100)),
                    false,
                    TruncationDirection::Right,
                    None
                )
                .await
                .is_err());
        });
    }

    fn metadata_tokenizer() -> Tokenizer {
        let mut tokenizer = crate::fast_tokenization::test_tokenizer();
        tokenizer.add_special_tokens(&[
            tokenizers::AddedToken::from("<s>", true),
            tokenizers::AddedToken::from("</s>", true),
        ]);
        tokenizer.with_post_processor(Some(
            tokenizers::processors::template::TemplateProcessing::builder()
                .try_single("<s>:0 $A:0 </s>:0")
                .unwrap()
                .try_pair("<s>:0 $A:0 </s>:0 $B:1 </s>:1")
                .unwrap()
                .special_tokens(vec![("<s>", 256), ("</s>", 257)])
                .build()
                .unwrap(),
        ));
        tokenizer
    }

    fn assert_encoding_ids(actual: &ValidEncoding, expected: &RawEncoding) {
        assert_eq!(actual.input_ids, expected.get_ids());
        assert_eq!(actual.token_type_ids, expected.get_type_ids());
        assert_eq!(
            actual.position_ids,
            (2..2 + expected.get_ids().len() as u32).collect::<Vec<_>>()
        );
    }

    fn hf_single_encoding(
        hf: &Tokenizer,
        text: &str,
        prompt: Option<&str>,
        add_special_tokens: bool,
        truncation: Option<(usize, TruncationDirection)>,
    ) -> RawEncoding {
        let mut hf = hf.clone();
        hf.with_truncation(truncation.map(|(max_length, direction)| TruncationParams {
            direction,
            max_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        }))
        .unwrap();
        let text = match prompt {
            Some(prompt) => format!("{prompt}{text}"),
            None => text.to_owned(),
        };
        hf.encode(text, add_special_tokens).unwrap()
    }

    fn hf_pair_encoding(
        hf: &Tokenizer,
        pair: (&str, &str),
        add_special_tokens: bool,
        truncation: Option<(usize, TruncationDirection)>,
    ) -> RawEncoding {
        let mut hf = hf.clone();
        hf.with_truncation(truncation.map(|(max_length, direction)| TruncationParams {
            direction,
            max_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        }))
        .unwrap();
        hf.encode(pair, add_special_tokens).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn tokenization_with_seed_and_fast(
        workers: usize,
        queue_capacity: usize,
        tokenizer: Tokenizer,
        seed: Option<Arc<Tokenizer>>,
        fast_tokenizer: crate::fast_tokenization::BackgroundFastTokenizer,
        max_input_length: usize,
        position_offset: usize,
        default_prompt: Option<String>,
        prompts: Option<HashMap<String, String>>,
    ) -> Tokenization {
        let workers = workers.clamp(1, MAX_TOKENIZATION_WORKERS);
        let (sender, receiver) = async_channel::bounded(queue_capacity);
        let config = TokenizerWorkerConfig {
            workers,
            receiver,
            tokenizer,
            accelerate_pooled_text: false,
            max_input_length,
            position_offset,
            default_prompt,
            prompts,
            chat: None,
        };
        let supervisor = std::thread::Builder::new()
            .name("test-tokenization-supervisor".into())
            .spawn(move || {
                run_worker_pool(config, seed, fast_tokenizer);
            })
            .unwrap();
        Tokenization {
            workers: Arc::new(TokenizerWorkers {
                sender: sender.clone(),
                state: std::sync::Mutex::new(WorkerState::Started(Some(supervisor))),
                started: AtomicBool::new(true),
                #[cfg(test)]
                supervisor_starts: std::sync::atomic::AtomicUsize::new(0),
            }),
            sender,
            multimodal: None,
            mistral3: None,
            embedding_gemma2: None,
            token_input_vocab_size: None,
            max_input_length,
            position_offset,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tokenization_with_fast(
        workers: usize,
        queue_capacity: usize,
        tokenizer: Tokenizer,
        max_input_length: usize,
        position_offset: usize,
        default_prompt: Option<String>,
        prompts: Option<HashMap<String, String>>,
        fast_tokenizer: crate::fast_tokenization::BackgroundFastTokenizer,
    ) -> Tokenization {
        let workers = workers.clamp(1, MAX_TOKENIZATION_WORKERS);
        let seed = if workers > 1 {
            Some(Arc::new(tokenizer.clone()))
        } else {
            None
        };
        tokenization_with_seed_and_fast(
            workers,
            queue_capacity,
            tokenizer,
            seed,
            fast_tokenizer,
            max_input_length,
            position_offset,
            default_prompt,
            prompts,
        )
    }

    struct ReleaseOnDrop(Option<std::sync::mpsc::Sender<()>>);

    impl ReleaseOnDrop {
        fn release(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            self.release();
        }
    }

    fn lazy_metadata_tokenizer(
        workers: usize,
        queue_capacity: usize,
        tokenizer: Tokenizer,
    ) -> Tokenization {
        Tokenization::with_lazy_queue_capacity(
            workers,
            NonZeroUsize::new(queue_capacity).unwrap(),
            tokenizer,
            64,
            2,
            None,
            None,
            None,
        )
    }

    fn supervisor_starts(tokenization: &Tokenization) -> usize {
        tokenization
            .workers
            .supervisor_starts
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[test]
    fn decision_selected_worker_keeps_metadata_tokenize_and_decode() {
        let hf = metadata_tokenizer();
        let expected = hf.clone();
        let tokenization = Tokenization::with_queue_capacity_and_pooled_text_acceleration(
            2,
            NonZeroUsize::new(4).unwrap(),
            hf,
            64,
            2,
            None,
            None,
            None,
            false,
        );
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let text = "hello 東京 👩🏽‍💻 <s> longer text";
            let (rendered, encoding) = tokenization
                .tokenize(EncodingInput::Single(text.into()), true, None)
                .await
                .unwrap();
            let expected_encoding = expected.encode(text, true).unwrap();
            assert_eq!(rendered.as_deref(), Some(text));
            assert_eq!(encoding.get_ids(), expected_encoding.get_ids());
            assert_eq!(encoding.get_type_ids(), expected_encoding.get_type_ids());
            assert_eq!(encoding.get_offsets(), expected_encoding.get_offsets());
            assert!(!encoding.get_tokens().is_empty());

            let decoded = tokenization
                .decode(encoding.get_ids().to_vec(), false)
                .await
                .unwrap();
            assert_eq!(
                decoded,
                expected.decode(encoding.get_ids(), false).unwrap(),
                "decision tokenization workers must retain HF decode behavior"
            );
            assert!(decoded.contains("hello 東京"));
        });
    }

    #[test]
    fn pooled_text_acceleration_choice_preserves_embedding_and_classifier_ids() {
        let hf = metadata_tokenizer();
        let expected = hf.clone();
        let accelerated = Tokenization::with_queue_capacity_and_pooled_text_acceleration(
            1,
            NonZeroUsize::new(8).unwrap(),
            hf.clone(),
            64,
            2,
            None,
            None,
            None,
            true,
        );
        let unaccelerated = Tokenization::with_queue_capacity_and_pooled_text_acceleration(
            1,
            NonZeroUsize::new(8).unwrap(),
            hf,
            64,
            2,
            None,
            None,
            None,
            false,
        );
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let single = "hello 東京 👩🏽‍💻 longer text";
            let pair = ("query 東京", "a longer document 👩🏽‍💻");
            for use_pair in [false, true] {
                let input = || {
                    if use_pair {
                        EncodingInput::Dual(pair.0.into(), pair.1.into())
                    } else {
                        EncodingInput::Single(single.into())
                    }
                };
                let expected_encoding = if use_pair {
                    expected.encode(pair, true).unwrap()
                } else {
                    expected.encode(single, true).unwrap()
                };

                let accelerated_embedding = accelerated
                    .encode_embedding(input(), false, TruncationDirection::Right, None)
                    .await
                    .unwrap();
                let unaccelerated_embedding = unaccelerated
                    .encode_embedding(input(), false, TruncationDirection::Right, None)
                    .await
                    .unwrap();
                assert_encoding_ids(&accelerated_embedding, &expected_encoding);
                assert_encoding_ids(&unaccelerated_embedding, &expected_encoding);
                assert!(accelerated_embedding.tokens.is_empty());
                assert!(unaccelerated_embedding.offsets.is_empty());

                let accelerated_classification = accelerated
                    .encode_classification(input(), false, TruncationDirection::Right)
                    .await
                    .unwrap();
                let unaccelerated_classification = unaccelerated
                    .encode_classification(input(), false, TruncationDirection::Right)
                    .await
                    .unwrap();
                assert_encoding_ids(&accelerated_classification, &expected_encoding);
                assert_encoding_ids(&unaccelerated_classification, &expected_encoding);
                assert!(accelerated_classification.tokens.is_empty());
                assert!(unaccelerated_classification.offsets.is_empty());
            }
        });
    }

    #[test]
    fn lazy_workers_start_once_on_first_metadata_request() {
        let hf = metadata_tokenizer();
        let text = "hello 東京 👩‍💻 <s> longer text";
        let ids = hf.encode(text, true).unwrap().get_ids().to_vec();
        for operation in ["tokenize", "decode", "encode"] {
            let expected = hf.clone();
            let tokenization = lazy_metadata_tokenizer(2, 4, hf.clone());
            assert_eq!(supervisor_starts(&tokenization), 0);
            assert!(matches!(
                &*tokenization.workers.state.lock().unwrap(),
                WorkerState::Idle(_)
            ));
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                match operation {
                    "tokenize" => {
                        let (rendered, actual) = tokenization
                            .tokenize(EncodingInput::Single(text.into()), true, None)
                            .await
                            .unwrap();
                        let expected_encoding = expected.encode(text, true).unwrap();
                        assert_eq!(rendered.as_deref(), Some(text));
                        assert_eq!(actual.get_ids(), expected_encoding.get_ids());
                        assert_eq!(actual.get_type_ids(), expected_encoding.get_type_ids());
                        assert_eq!(actual.get_offsets(), expected_encoding.get_offsets());
                    }
                    "decode" => {
                        let actual = tokenization.decode(ids.clone(), false).await.unwrap();
                        assert_eq!(actual, expected.decode(ids.as_slice(), false).unwrap());
                    }
                    "encode" => {
                        let actual = tokenization
                            .encode(
                                EncodingInput::Single(text.into()),
                                true,
                                TruncationDirection::Right,
                                None,
                            )
                            .await
                            .unwrap();
                        assert_encoding_ids(&actual, &expected.encode(text, true).unwrap());
                        assert!(!actual.tokens.is_empty());
                        assert!(!actual.offsets.is_empty());
                    }
                    _ => unreachable!(),
                }
            });
            assert_eq!(supervisor_starts(&tokenization), 1);
            assert!(matches!(
                &*tokenization.workers.state.lock().unwrap(),
                WorkerState::Started(_)
            ));
        }
    }

    #[test]
    fn concurrent_first_lazy_requests_start_one_pool_and_preserve_outputs() {
        let hf = metadata_tokenizer();
        let tokenization = lazy_metadata_tokenizer(4, 2, hf.clone());
        assert_eq!(supervisor_starts(&tokenization), 0);
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut tasks = tokio::task::JoinSet::new();
            for i in 0..64 {
                let tokenization = tokenization.clone();
                let expected = hf.clone();
                tasks.spawn(async move {
                    let text = format!("request {i} 東京 👩‍💻 <s>");
                    let expected_encoding = expected.encode(text.as_str(), true).unwrap();
                    let (rendered, actual) = tokenization
                        .tokenize(EncodingInput::Single(text.clone()), true, None)
                        .await
                        .unwrap();
                    assert_eq!(rendered.as_deref(), Some(text.as_str()));
                    assert_eq!(actual.get_ids(), expected_encoding.get_ids());
                    assert_eq!(actual.get_type_ids(), expected_encoding.get_type_ids());
                    assert_eq!(actual.get_offsets(), expected_encoding.get_offsets());
                    let decoded = tokenization
                        .decode(actual.get_ids().to_vec(), false)
                        .await
                        .unwrap();
                    assert_eq!(decoded, expected.decode(actual.get_ids(), false).unwrap());
                    let encoded = tokenization
                        .encode(
                            EncodingInput::Single(text),
                            true,
                            TruncationDirection::Right,
                            None,
                        )
                        .await
                        .unwrap();
                    assert_eq!(encoded.input_ids, expected_encoding.get_ids());
                    assert_eq!(encoded.token_type_ids, expected_encoding.get_type_ids());
                });
            }
            while let Some(result) = tasks.join_next().await {
                result.unwrap();
            }
        });
        assert_eq!(supervisor_starts(&tokenization), 1);
    }

    #[test]
    fn shutdown_before_first_use_starts_no_workers_and_rejects_requests() {
        let tokenization = lazy_metadata_tokenizer(2, 4, metadata_tokenizer());
        assert_eq!(supervisor_starts(&tokenization), 0);
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            tokenization.shutdown().await;
            assert!(matches!(
                tokenization
                    .tokenize(EncodingInput::Single("hello".into()), true, None)
                    .await,
                Err(TextEmbeddingsError::Backend(BackendError::Inference(_)))
            ));
            assert!(matches!(
                tokenization.decode(vec![1, 2], false).await,
                Err(TextEmbeddingsError::Backend(BackendError::Inference(_)))
            ));
            assert!(matches!(
                tokenization
                    .encode(
                        EncodingInput::Single("hello".into()),
                        true,
                        TruncationDirection::Right,
                        None,
                    )
                    .await,
                Err(TextEmbeddingsError::Backend(BackendError::Inference(_)))
            ));
        });
        assert_eq!(supervisor_starts(&tokenization), 0);
        assert!(matches!(
            &*tokenization.workers.state.lock().unwrap(),
            WorkerState::Shutdown
        ));
    }

    #[test]
    fn shutdown_and_first_request_race_never_restarts_workers() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        for _ in 0..16 {
            let tokenization = lazy_metadata_tokenizer(2, 2, metadata_tokenizer());
            let (first, after) = runtime.block_on(async {
                let request = tokenization.clone();
                let first = tokio::join!(
                    request.tokenize(EncodingInput::Single("hello".into()), true, None),
                    tokenization.shutdown(),
                )
                .0;
                let after = tokenization
                    .tokenize(EncodingInput::Single("hello".into()), true, None)
                    .await;
                (first, after)
            });
            assert!(matches!(
                first,
                Ok(_) | Err(TextEmbeddingsError::Backend(BackendError::Inference(_)))
            ));
            assert!(matches!(
                after,
                Err(TextEmbeddingsError::Backend(BackendError::Inference(_)))
            ));
            assert!(supervisor_starts(&tokenization) <= 1);
            assert!(matches!(
                &*tokenization.workers.state.lock().unwrap(),
                WorkerState::Shutdown
            ));
        }
    }

    #[test]
    fn embedding_and_classifier_finish_while_fast_tokenizer_initializes() {
        let hf = metadata_tokenizer();
        let expected = hf.clone();
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let fast = crate::fast_tokenization::BackgroundFastTokenizer::new_gated(
            Arc::new(hf.clone()),
            started_tx,
            release_rx,
        );
        let probe = fast.clone();
        let tokenization = tokenization_with_fast(
            4,
            8,
            hf,
            8,
            2,
            None,
            Some(HashMap::from([("query".into(), "query: ".into())])),
            fast,
        );
        started_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert!(probe.get().is_none());
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let text = "hello 東京 👩‍💻 <s> longer text";
            let pair = ("query 東京", "a longer document 👩‍💻");
            for pass in 0..2 {
                if pass == 1 {
                    release_tx.send(()).unwrap();
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                    while probe.get().is_none() {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "fast tokenizer did not publish"
                        );
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
                for direction in [TruncationDirection::Left, TruncationDirection::Right] {
                    let expected_embedding = hf_single_encoding(
                        &expected,
                        text,
                        Some("query: "),
                        true,
                        Some((8, direction)),
                    );
                    let actual = tokio::time::timeout(
                        std::time::Duration::from_secs(2),
                        tokenization.encode_embedding(
                            EncodingInput::Single(text.into()),
                            true,
                            direction,
                            Some("query".into()),
                        ),
                    )
                    .await
                    .expect("embedding request blocked on fast initialization")
                    .unwrap();
                    assert_encoding_ids(&actual, &expected_embedding);
                    assert!(actual.tokens.is_empty() && actual.offsets.is_empty());

                    let expected_pair =
                        hf_pair_encoding(&expected, pair, true, Some((8, direction)));
                    let actual = tokio::time::timeout(
                        std::time::Duration::from_secs(2),
                        tokenization.encode_embedding(
                            EncodingInput::Dual(pair.0.into(), pair.1.into()),
                            true,
                            direction,
                            None,
                        ),
                    )
                    .await
                    .expect("dual embedding blocked on fast initialization")
                    .unwrap();
                    assert_encoding_ids(&actual, &expected_pair);
                    assert!(actual.tokens.is_empty() && actual.offsets.is_empty());

                    let expected_classification =
                        hf_single_encoding(&expected, text, None, true, Some((8, direction)));
                    let actual = tokio::time::timeout(
                        std::time::Duration::from_secs(2),
                        tokenization.encode_classification(
                            EncodingInput::Single(text.into()),
                            true,
                            direction,
                        ),
                    )
                    .await
                    .expect("classification request blocked on fast initialization")
                    .unwrap();
                    assert_encoding_ids(&actual, &expected_classification);
                    assert!(actual.tokens.is_empty() && actual.offsets.is_empty());

                    let expected_classification_pair =
                        hf_pair_encoding(&expected, pair, true, Some((8, direction)));
                    let actual = tokio::time::timeout(
                        std::time::Duration::from_secs(2),
                        tokenization.encode_classification(
                            EncodingInput::Dual(pair.0.into(), pair.1.into()),
                            true,
                            direction,
                        ),
                    )
                    .await
                    .expect("dual classification blocked on fast initialization")
                    .unwrap();
                    assert_encoding_ids(&actual, &expected_classification_pair);
                    assert!(actual.tokens.is_empty() && actual.offsets.is_empty());
                }

                let expected_rendered = format!("query: {text}");
                let expected_metadata =
                    hf_single_encoding(&expected, text, Some("query: "), true, None);
                let (rendered, actual) = tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    tokenization.tokenize(
                        EncodingInput::Single(text.into()),
                        true,
                        Some("query".into()),
                    ),
                )
                .await
                .expect("metadata tokenization blocked on fast initialization")
                .unwrap();
                assert_eq!(rendered.as_deref(), Some(expected_rendered.as_str()));
                assert_eq!(actual.get_ids(), expected_metadata.get_ids());
                assert_eq!(actual.get_type_ids(), expected_metadata.get_type_ids());
                assert_eq!(actual.get_offsets(), expected_metadata.get_offsets());
                assert_eq!(actual.get_tokens(), expected_metadata.get_tokens());
                let decoded = tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    tokenization.decode(actual.get_ids().to_vec(), false),
                )
                .await
                .expect("decode blocked on fast initialization")
                .unwrap();
                assert_eq!(decoded, expected.decode(actual.get_ids(), false).unwrap());
            }
        });
        assert!(probe.get().is_some());
    }

    #[test]
    fn shutdown_waits_for_blocked_fast_tokenizer_builder() {
        let hf = metadata_tokenizer();
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let fast = crate::fast_tokenization::BackgroundFastTokenizer::new_gated(
            Arc::new(hf.clone()),
            started_tx,
            release_rx,
        );
        let tokenization = tokenization_with_fast(1, 4, hf, 64, 0, None, None, fast);
        started_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let shutdown_tokenization = tokenization.clone();
            let shutdown = tokio::spawn(async move { shutdown_tokenization.shutdown().await });
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            assert!(!shutdown.is_finished());
            release_tx.send(()).unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(5), shutdown)
                .await
                .expect("shutdown blocked after builder release")
                .unwrap();
        });
    }

    #[test]
    fn single_worker_releases_unused_fast_tokenizer_seed_after_publication() {
        let hf = metadata_tokenizer();
        let expected = hf.clone();
        let template = Arc::new(hf.clone());
        let template_weak = Arc::downgrade(&template);
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let fast = crate::fast_tokenization::BackgroundFastTokenizer::new_gated(
            Arc::clone(&template),
            started_tx,
            release_rx,
        );
        let probe = fast.clone();
        let tokenization =
            tokenization_with_seed_and_fast(1, 4, hf, Some(template), fast, 64, 2, None, None);
        let mut release = ReleaseOnDrop(Some(release_tx));
        started_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("background tokenizer builder did not start");
        assert!(probe.get().is_none());

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let text = "hello 東京 👩‍💻 <s> longer text";
        let expected_encoding = hf_single_encoding(&expected, text, None, true, None);
        runtime.block_on(async {
            let actual = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                tokenization.encode_embedding(
                    EncodingInput::Single(text.into()),
                    false,
                    TruncationDirection::Right,
                    None,
                ),
            )
            .await
            .expect("request blocked on background tokenizer initialization")
            .unwrap();
            assert_encoding_ids(&actual, &expected_encoding);
            assert!(actual.tokens.is_empty() && actual.offsets.is_empty());
        });

        release.release();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while probe.get().is_none() {
            assert!(
                std::time::Instant::now() < deadline,
                "background tokenizer did not publish"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        while template_weak.upgrade().is_some() {
            assert!(
                std::time::Instant::now() < deadline,
                "single-worker pool retained the unused HF seed template"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        runtime.block_on(async {
            let actual = tokenization
                .encode_embedding(
                    EncodingInput::Single(text.into()),
                    false,
                    TruncationDirection::Right,
                    None,
                )
                .await
                .unwrap();
            assert_encoding_ids(&actual, &expected_encoding);
            assert!(actual.tokens.is_empty() && actual.offsets.is_empty());
            tokio::time::timeout(std::time::Duration::from_secs(5), tokenization.shutdown())
                .await
                .expect("tokenization shutdown timed out");
        });
    }

    #[test]
    fn concurrent_requests_preserve_fast_and_fallback_ids() {
        for capacity in [1, DEFAULT_TOKENIZATION_QUEUE_CAPACITY] {
            let tokenizer = Tokenization::with_queue_capacity(
                4,
                NonZeroUsize::new(capacity).unwrap(),
                crate::fast_tokenization::test_tokenizer(),
                256,
                0,
                None,
                None,
                None,
            );
            assert_eq!(tokenizer.sender.capacity(), Some(capacity));
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let mut tasks = tokio::task::JoinSet::new();
                for i in 0..256 {
                    let tokenizer = tokenizer.clone();
                    tasks.spawn(async move {
                        let pair = (format!("query {i}"), "東京 passage".to_string());
                        let expected = tokenizer
                            .encode(pair.clone().into(), true, TruncationDirection::Right, None)
                            .await
                            .unwrap();
                        let actual = tokenizer
                            .encode_embedding(pair.into(), true, TruncationDirection::Right, None)
                            .await
                            .unwrap();
                        assert_eq!(actual.input_ids, expected.input_ids);
                        assert_eq!(actual.token_type_ids, expected.token_type_ids);
                        let text = format!("request {i} 東京 👩🏽‍💻");
                        let expected = tokenizer
                            .encode(text.clone().into(), true, TruncationDirection::Left, None)
                            .await
                            .unwrap();
                        let actual = tokenizer
                            .encode_embedding(text.into(), true, TruncationDirection::Left, None)
                            .await
                            .unwrap();
                        assert_eq!(actual.input_ids, expected.input_ids);
                        assert_eq!(actual.token_type_ids, expected.token_type_ids);
                    });
                }
                while let Some(result) = tasks.join_next().await {
                    result.unwrap();
                }
            });
        }
    }
}
