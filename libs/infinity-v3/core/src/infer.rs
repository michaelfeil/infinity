use crate::backend_pool::BackendPool;
use crate::queue_actor::{prune_canceled_batch, Entry, Metadata, NextBatch, Queue};
use crate::tokenization::{EncodingInput, RawEncoding, Tokenization};
use crate::TextEmbeddingsError;
use infinity_v3_backend::result_memory::{retain_for_request, HasResultMemory, ResultLease};
use infinity_v3_backend::{Backend, BackendError, Embedding, ModelType};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokenizers::TruncationDirection;
use tokio::sync::{mpsc, oneshot, Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tracing::instrument;

/// Request entry point and shared lifetime of the inference workers.
///
/// Clones share the queue, replica pool and shutdown state. `with_pool` starts
/// one batch preparer and one submitter per replica, with a single shared slot
/// for a prepared batch. Per-request replies travel through `Metadata` channels.
#[derive(Debug, Clone)]
pub struct Infer {
    model_name: metrics::SharedString,
    tokenization: Tokenization,
    queue: Queue,
    /// Shared notify
    notify_batching_task: Arc<Notify>,
    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,
    backend: BackendPool,
    _lifetime: Arc<PipelineLifetime>,
}

// Every admitted input must release its preparation count and wake the worker,
// including failed or canceled tokenization. Waiting for all client inputs can
// deadlock when the client batch is larger than the admission semaphore.
struct BatchPreparation<'a> {
    remaining: Option<&'a AtomicUsize>,
    notify: &'a Notify,
}

impl Drop for BatchPreparation<'_> {
    fn drop(&mut self) {
        if let Some(remaining) = self.remaining {
            remaining.fetch_sub(1, Ordering::SeqCst);
        }
        self.notify.notify_one();
    }
}

impl Infer {
    pub async fn decide(
        &self,
        encoding: crate::tokenization::ValidEncoding,
        input: infinity_v3_backend::DecisionInput,
        tokenization: Duration,
        batch_counter: Arc<AtomicUsize>,
    ) -> Result<DecisionInferResponse, TextEmbeddingsError> {
        let preparation = BatchPreparation {
            remaining: Some(batch_counter.as_ref()),
            notify: &self.notify_batching_task,
        };
        if self.backend.model_type != ModelType::Decision {
            return Err(TextEmbeddingsError::Validation(
                "Loaded model does not support typed decisions".into(),
            ));
        }
        let (response_tx, response_rx) = oneshot::channel();
        self.queue
            .append(Entry {
                metadata: Metadata {
                    input_memory: None,
                    client_batch: Some(batch_counter.clone()),
                    response_tx,
                    tokenization,
                    queue_time: Instant::now(),
                    prompt_tokens: encoding.input_ids.len(),
                    pooling: false,
                    token_classification: false,
                    decision: Some(input),
                },
                encoding,
            })
            .await;
        drop(preparation);
        match response_rx.await.map_err(|_| BackendError::Unhealthy)?? {
            InferResult::Decision(output) => {
                retain_for_request(output.metadata.memory.clone());
                Ok(output)
            }
            _ => Err(BackendError::Inference("Unexpected decision response".into()).into()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenization: Tokenization,
        queue: Queue,
        max_concurrent_requests: usize,
        backend: Backend,
    ) -> Self {
        Self::with_pool(
            tokenization,
            queue,
            max_concurrent_requests,
            BackendPool::new(vec![backend]).expect("single backend is compatible"),
        )
    }

    pub fn with_pool(
        tokenization: Tokenization,
        queue: Queue,
        max_concurrent_requests: usize,
        backend: BackendPool,
    ) -> Self {
        let notify_batching_task = Arc::new(Notify::new());
        let (sender, receiver) = mpsc::channel(1);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut tasks = vec![tokio::spawn(batching_task(
            backend.clone(),
            queue.clone(),
            notify_batching_task.clone(),
            sender,
        ))];
        for replica in 0..backend.len() {
            tasks.push(tokio::spawn(backend_task(
                backend.clone(),
                queue.clone(),
                receiver.clone(),
                replica,
            )));
        }
        let lifetime = Arc::new(PipelineLifetime {
            pool: backend.clone(),
            queue: queue.clone(),
            tasks: Mutex::new(tasks),
            shutdown: Mutex::new(()),
        });
        Self {
            model_name: backend.model_label(),
            tokenization,
            queue,
            notify_batching_task,
            limit_concurrent_requests: Arc::new(Semaphore::new(max_concurrent_requests)),
            backend,
            _lifetime: lifetime,
        }
    }

    #[instrument(skip(self, inputs))]
    pub async fn tokenize<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        add_special_tokens: bool,
        prompt_name: Option<String>,
    ) -> Result<(Option<String>, RawEncoding), TextEmbeddingsError> {
        self.tokenization
            .tokenize(inputs.into(), add_special_tokens, prompt_name)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "tokenization");
                counter.increment(1);
                log_request_error(&err);
                err
            })
    }

    #[instrument(skip(self, ids))]
    pub async fn decode(
        &self,
        ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> Result<String, TextEmbeddingsError> {
        self.tokenization
            .decode(ids, skip_special_tokens)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "tokenization");
                counter.increment(1);
                log_request_error(&err);
                err
            })
    }

    #[instrument(skip(self))]
    pub fn try_acquire_permit(&self) -> Result<OwnedSemaphorePermit, TextEmbeddingsError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        self.clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "overloaded");
                counter.increment(1);
                tracing::warn!("{err}");
                TextEmbeddingsError::from(err)
            })
    }

    #[instrument(skip(self))]
    pub async fn acquire_permit(&self) -> OwnedSemaphorePermit {
        // Limit concurrent requests by acquiring a permit from the semaphore
        self.clone()
            .limit_concurrent_requests
            .acquire_owned()
            .await
            .expect("Semaphore has been closed. This is a bug.")
    }

    #[instrument(skip(self, inputs, permit))]
    pub async fn embed_all<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<AllEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        if self.is_splade() {
            let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "model_type");
            counter.increment(1);
            let message = "`embed_all` is not available for SPLADE models".to_string();
            tracing::warn!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let results = self
            .embed(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                false,
                &start_time,
                permit,
                batch_counter,
            )
            .await?;

        let InferResult::AllEmbedding(response) = results else {
            panic!("unexpected enum variant")
        };

        let total_time = start_time.elapsed();

        metrics::counter!("infinity_embed_success", "model" => self.model_name.clone())
            .increment(1);
        metrics::histogram!("infinity_embed_duration", "model" => self.model_name.clone())
            .record(total_time.as_secs_f64());
        metrics::histogram!("infinity_embed_tokenization_duration", "model" => self.model_name.clone())
            .record(response.metadata.tokenization.as_secs_f64());
        metrics::histogram!("infinity_embed_queue_duration", "model" => self.model_name.clone())
            .record(response.metadata.queue.as_secs_f64());
        metrics::histogram!("infinity_embed_inference_duration", "model" => self.model_name.clone())
            .record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[instrument(skip(self, inputs, permit))]
    pub async fn embed_sparse<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<PooledEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        if !self.is_splade() {
            let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "model_type");
            counter.increment(1);
            let message = "Model is not an embedding model with SPLADE pooling".to_string();
            tracing::warn!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let results = self
            .embed(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                true,
                &start_time,
                permit,
                batch_counter,
            )
            .await?;

        let InferResult::PooledEmbedding(response) = results else {
            panic!("unexpected enum variant")
        };

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        let counter =
            metrics::counter!("infinity_embed_success", "model" => self.model_name.clone());
        counter.increment(1);
        let histogram =
            metrics::histogram!("infinity_embed_duration", "model" => self.model_name.clone());
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("infinity_embed_tokenization_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("infinity_embed_queue_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("infinity_embed_inference_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self, inputs, permit))]
    pub async fn embed_pooled<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        normalize: bool,
        dimensions: Option<usize>,
        permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<PooledEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        if self.is_splade() && normalize {
            let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "model_type");
            counter.increment(1);

            let message = "`normalize` is not available for SPLADE models".to_string();
            tracing::warn!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        if let Some(dimensions) = dimensions {
            if dimensions == 0 {
                metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "validation").increment(1);
                let message = "`dimensions` should be positive".to_string();
                tracing::warn!("{message}");
                return Err(TextEmbeddingsError::Validation(message));
            }
        }

        let results = self
            .embed(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                true,
                &start_time,
                permit,
                batch_counter,
            )
            .await?;

        let InferResult::PooledEmbedding(mut response) = results else {
            panic!("unexpected enum variant")
        };

        if let Some(mrl_dimensions) = dimensions {
            if mrl_dimensions > response.results.len() {
                metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "validation").increment(1);

                let message =
                    "`dimensions` should be smaller than the maximum embedding dimension."
                        .to_string();
                tracing::warn!("{message}");

                return Err(TextEmbeddingsError::Validation(message));
            }

            response.results.truncate(mrl_dimensions);
        }

        if normalize {
            // Normalize embedding
            // TODO: Should this be normalized with a background thread etc. instead of doing it synchronously here?
            let scale = (1.0
                / response
                    .results
                    .iter()
                    .map(|v| {
                        let v = *v as f64;
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt()) as f32;
            for v in response.results.iter_mut() {
                *v *= scale;
            }
        }

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        let counter =
            metrics::counter!("infinity_embed_success", "model" => self.model_name.clone());
        counter.increment(1);
        let histogram =
            metrics::histogram!("infinity_embed_duration", "model" => self.model_name.clone());
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("infinity_embed_tokenization_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("infinity_embed_queue_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("infinity_embed_inference_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[allow(clippy::too_many_arguments)]
    async fn embed<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        pooling: bool,
        start_time: &Instant,
        _permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<InferResult, TextEmbeddingsError> {
        let preparation = BatchPreparation {
            remaining: batch_counter.as_deref(),
            notify: &self.notify_batching_task,
        };
        if !matches!(self.backend.model_type, ModelType::Embedding(_)) {
            let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "model_type");
            counter.increment(1);
            let message = "Model is not an embedding model".to_string();
            tracing::warn!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let counter = metrics::counter!("infinity_embed_count", "model" => self.model_name.clone());
        counter.increment(1);

        // Tokenization
        let encoding = self
            .tokenization
            .encode_embedding(inputs.into(), truncate, truncation_direction, prompt_name)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "tokenization");
                counter.increment(1);
                log_request_error(&err);
                err
            })?;

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = oneshot::channel();

        // Append the request to the queue
        self.queue
            .append(Entry {
                metadata: Metadata {
                    input_memory: None,
                    client_batch: batch_counter.clone(),
                    response_tx,
                    tokenization: start_time.elapsed(),
                    queue_time: Instant::now(),
                    prompt_tokens: encoding.input_ids.len(),
                    pooling,
                    token_classification: false,
                    decision: None,
                },
                encoding,
            })
            .await;

        drop(preparation);

        let response = response_rx
            .await
            .expect(
                "Infer batching task dropped the sender without sending a response. This is a bug.",
            )
            .map_err(|err| {
                let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "inference");
                counter.increment(1);
                let err = TextEmbeddingsError::from(err);
                log_request_error(&err);
                err
            })?;

        response.retain_for_request();
        Ok(response)
    }

    #[instrument(skip(self, inputs, _permit))]
    pub async fn predict<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        raw_scores: bool,
        _permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<ClassificationInferResponse, TextEmbeddingsError> {
        let preparation = BatchPreparation {
            remaining: batch_counter.as_deref(),
            notify: &self.notify_batching_task,
        };
        if !self.is_classifier() {
            let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "model_type");
            counter.increment(1);
            let message = "Model is not a classifier model".to_string();
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let start_time = Instant::now();
        let counter =
            metrics::counter!("infinity_predict_count", "model" => self.model_name.clone());
        counter.increment(1);

        // Tokenization
        let encoding = self
            .tokenization
            .encode(inputs.into(), truncate, truncation_direction, None)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "tokenization");
                counter.increment(1);
                log_request_error(&err);
                err
            })?;

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = oneshot::channel();

        // Append the request to the queue
        self.queue
            .append(Entry {
                metadata: Metadata {
                    input_memory: None,
                    client_batch: batch_counter.clone(),
                    response_tx,
                    tokenization: start_time.elapsed(),
                    queue_time: Instant::now(),
                    prompt_tokens: encoding.input_ids.len(),
                    pooling: true,
                    token_classification: false,
                    decision: None,
                },
                encoding,
            })
            .await;

        drop(preparation);

        let response = response_rx
            .await
            .expect(
                "Infer batching task dropped the sender without sending a response. This is a bug.",
            )
            .map_err(|err| {
                let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "inference");
                counter.increment(1);
                let err = TextEmbeddingsError::from(err);
                log_request_error(&err);
                err
            })?;

        response.retain_for_request();
        let InferResult::Classification(mut response) = response else {
            panic!("unexpected enum variant")
        };

        if !raw_scores {
            // Softmax
            if response.results.len() > 1 {
                let max = response
                    .results
                    .iter()
                    .copied()
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap();

                let mut den = 0.0;
                for v in response.results.iter_mut() {
                    *v = (*v - max).exp();
                    den += *v;
                }
                for v in response.results.iter_mut() {
                    *v /= den;
                }
            }
            // Sigmoid
            else {
                response.results[0] = 1.0 / (1.0 + (-response.results[0]).exp());
            }
        }

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        let counter =
            metrics::counter!("infinity_predict_success", "model" => self.model_name.clone());
        counter.increment(1);
        let histogram =
            metrics::histogram!("infinity_predict_duration", "model" => self.model_name.clone());
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("infinity_predict_tokenization_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("infinity_predict_queue_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("infinity_predict_inference_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[instrument(skip(self, inputs, _permit))]
    pub async fn predict_tokens<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        raw_scores: bool,
        _permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<TokenClassificationInferResponse, TextEmbeddingsError> {
        let preparation = BatchPreparation {
            remaining: batch_counter.as_deref(),
            notify: &self.notify_batching_task,
        };
        if !self.is_classifier() {
            let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "model_type");
            counter.increment(1);
            let message = "Model is not a classifier model".to_string();
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let start_time = Instant::now();
        let counter =
            metrics::counter!("infinity_predict_count", "model" => self.model_name.clone());
        counter.increment(1);

        let encoding = self
            .tokenization
            .encode(inputs.into(), truncate, truncation_direction, None)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "tokenization");
                counter.increment(1);
                log_request_error(&err);
                err
            })?;

        let (response_tx, response_rx) = oneshot::channel();

        self.queue
            .append(Entry {
                metadata: Metadata {
                    input_memory: None,
                    client_batch: batch_counter.clone(),
                    response_tx,
                    tokenization: start_time.elapsed(),
                    queue_time: Instant::now(),
                    prompt_tokens: encoding.input_ids.len(),
                    pooling: false,
                    token_classification: true,
                    decision: None,
                },
                encoding,
            })
            .await;

        drop(preparation);

        let response = response_rx
            .await
            .expect(
                "Infer batching task dropped the sender without sending a response. This is a bug.",
            )
            .map_err(|err| {
                let counter = metrics::counter!("infinity_request_failure", "model" => self.model_name.clone(), "err" => "inference");
                counter.increment(1);
                let err = TextEmbeddingsError::from(err);
                log_request_error(&err);
                err
            })?;

        response.retain_for_request();
        let InferResult::TokenClassification(mut response) = response else {
            panic!("unexpected enum variant")
        };

        if !raw_scores {
            for (_, _, scores, _, _) in response.results.iter_mut() {
                if scores.len() > 1 {
                    let max = scores
                        .iter()
                        .copied()
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap();

                    let mut den = 0.0;
                    for v in scores.iter_mut() {
                        *v = (*v - max).exp();
                        den += *v;
                    }
                    for v in scores.iter_mut() {
                        *v /= den;
                    }
                } else {
                    scores[0] = 1.0 / (1.0 + (-scores[0]).exp());
                }
            }
        }

        let total_time = start_time.elapsed();

        let counter =
            metrics::counter!("infinity_predict_success", "model" => self.model_name.clone());
        counter.increment(1);
        let histogram =
            metrics::histogram!("infinity_predict_duration", "model" => self.model_name.clone());
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("infinity_predict_tokenization_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("infinity_predict_queue_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("infinity_predict_inference_duration", "model" => self.model_name.clone());
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[instrument(skip(self))]
    pub fn is_available(&self) -> bool {
        !self.queue.is_closed() && self.backend.is_available()
    }

    pub fn is_classifier(&self) -> bool {
        matches!(self.backend.model_type, ModelType::Classifier)
    }

    #[instrument(skip(self))]
    pub fn is_splade(&self) -> bool {
        matches!(
            self.backend.model_type,
            ModelType::Embedding(infinity_v3_backend::Pool::Splade)
        )
    }

    /// Stop admission and await all batching, backend and tokenizer workers.
    pub async fn shutdown(&self) {
        let _guard = self._lifetime.shutdown.lock().await;
        self.limit_concurrent_requests.close();
        self.queue.close();
        self.backend.close();
        let tasks = std::mem::take(&mut *self._lifetime.tasks.lock().await);
        for task in tasks {
            let _ = task.await;
        }
        self.backend.shutdown().await;
        self.queue.shutdown().await;
        self.tokenization.shutdown().await;
    }

    #[instrument(skip(self))]
    pub async fn health(&self) -> bool {
        !self.queue.is_closed() && self.backend.health().await.is_ok()
    }
}

#[derive(Debug)]
struct PipelineLifetime {
    shutdown: Mutex<()>,
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    pool: BackendPool,
    queue: Queue,
}
impl Drop for PipelineLifetime {
    fn drop(&mut self) {
        self.pool.close();
        self.queue.close();
    }
}

struct DispatchedBatch(Option<NextBatch>);
impl Drop for DispatchedBatch {
    fn drop(&mut self) {
        if let Some((metadata, _)) = self.0.take() {
            for entry in metadata {
                let _ = entry.response_tx.send(Err(BackendError::Unhealthy));
            }
        }
    }
}

async fn batching_task(
    pool: BackendPool,
    queue: Queue,
    notify: Arc<Notify>,
    sender: mpsc::Sender<DispatchedBatch>,
) {
    let mut changed = pool.changes();
    loop {
        let arrival = notify.notified();
        tokio::pin!(arrival);
        arrival.as_mut().enable();
        // Reserve the only shared parking slot before preparing any batch.
        let permit = tokio::select! {
            permit = sender.reserve() => match permit { Ok(p) => p, Err(_) => break },
            _ = changed.changed() => { if pool.is_closed() { break; } continue; }
        };
        if pool.is_closed() {
            break;
        }
        let started = Instant::now();
        if let Some(batch) = queue.next_batch().await {
            pool.observe_preparation(started.elapsed());
            permit.send(DispatchedBatch(Some(batch)));
        } else {
            drop(permit);
            tokio::select! { _ = arrival => {}, _ = changed.changed() => {} }
        }
    }
    queue.close();
}

#[instrument(skip_all, fields(replica))]
async fn backend_task(
    backend: BackendPool,
    queue: Queue,
    receiver: Arc<Mutex<mpsc::Receiver<DispatchedBatch>>>,
    replica: usize,
) {
    let mut changed = backend.changes();
    let mut completions = tokio::task::JoinSet::new();
    loop {
        if backend.is_closed() {
            break;
        }
        let next = async { receiver.lock().await.recv().await };
        let mut dispatched = tokio::select! {
            batch = next => match batch { Some(b) => b, None => break },
            _ = changed.changed() => { if backend.is_closed() { break; } continue; }
        };
        let execution = match backend.acquire_replica(replica).await {
            Ok(execution) => execution,
            Err(_) => break,
        };
        let Some(batch) = prune_canceled_batch(dispatched.0.take().unwrap()) else {
            continue;
        };
        // Keep request metadata alive until completion, even after an early
        // model-lease release. Other backends keep their lease until results are ready.
        completions.spawn(complete_batch(execution, backend.model_type.clone(), batch));
        while completions.try_join_next().is_some() {}
    }
    while completions.join_next().await.is_some() {}
    queue.close();
}

async fn complete_batch(
    execution: crate::backend_pool::Execution,
    model_type: ModelType,
    (entries, mut batch): NextBatch,
) {
    match model_type {
        ModelType::Decision => {
            let inputs = entries
                .iter()
                .map(|m| m.decision.clone().expect("decision metadata missing"))
                .collect();
            let results = execution
                .decide(batch, inputs)
                .await
                .and_then(|(outputs, duration)| {
                    if outputs.len() != entries.len() {
                        return Err(BackendError::Inference(
                            "Decision output count does not match batch".into(),
                        ));
                    }
                    Ok((outputs.into_iter(), duration))
                });
            send_batch_results(entries, results, |outputs, _, metadata| {
                InferResult::Decision(DecisionInferResponse {
                    results: outputs.next().expect("decision output count was checked"),
                    metadata,
                })
            });
        }
        ModelType::Classifier if entries.iter().any(|m| m.token_classification) => {
            // Move source alignment into delivery; the backend only consumes numeric inputs.
            let input_ids = batch.input_ids.clone();
            let cumulative_seq_lengths = batch.cumulative_seq_lengths.clone();
            let mut tokens = std::mem::take(&mut batch.tokens);
            let offsets = std::mem::take(&mut batch.offsets);
            let results =
                execution
                    .predict_tokens(batch)
                    .await
                    .and_then(|(mut outputs, duration)| {
                        if let Some(memory) = &outputs.memory {
                            let rows = outputs
                                .values()
                                .try_fold(0usize, |n, rows| n.checked_add(rows.len()))
                                .ok_or_else(|| {
                                    BackendError::Inference("Token result count overflow".into())
                                })?;
                            let bytes = rows
                                .checked_mul(std::mem::size_of::<TokenPrediction>())
                                .ok_or_else(|| {
                                    BackendError::Inference("Token result memory overflow".into())
                                })?;
                            outputs.memory = Some(memory.reserve_more(bytes)?);
                        }
                        Ok((outputs, duration))
                    });
            // Token text/offset construction can be large; retain its blocking worker.
            tokio::task::spawn_blocking(move || {
                send_batch_results(entries, results, |predictions, index, metadata| {
                    let scores = predictions
                        .remove(&index)
                        .expect("prediction not found in results. This is a backend bug.");
                    let start = cumulative_seq_lengths[index] as usize;
                    let results = scores
                        .into_iter()
                        .enumerate()
                        .map(|(token_index, scores)| {
                            let global_index = start + token_index;
                            make_token_prediction(
                                global_index,
                                scores,
                                &input_ids,
                                &mut tokens,
                                &offsets,
                            )
                        })
                        .collect();
                    InferResult::TokenClassification(TokenClassificationInferResponse {
                        results,
                        metadata,
                    })
                });
            });
        }
        ModelType::Classifier => {
            send_batch_results(
                entries,
                execution.predict(batch).await,
                |predictions, index, metadata| {
                    InferResult::Classification(ClassificationInferResponse {
                        results: predictions
                            .remove(&index)
                            .expect("prediction not found in results. This is a backend bug."),
                        metadata,
                    })
                },
            );
        }
        ModelType::Embedding(_) => {
            send_batch_results(
                entries,
                execution.embed(batch).await,
                |embeddings, index, metadata| match embeddings
                    .remove(&index)
                    .expect("embedding not found in results. This is a backend bug.")
                {
                    Embedding::Pooled(results) => {
                        InferResult::PooledEmbedding(PooledEmbeddingsInferResponse {
                            results,
                            metadata,
                        })
                    }
                    Embedding::All(results) => {
                        InferResult::AllEmbedding(AllEmbeddingsInferResponse { results, metadata })
                    }
                },
            );
        }
    }
}

fn send_batch_results<T: HasResultMemory>(
    entries: Vec<Metadata>,
    results: Result<(T, Duration), BackendError>,
    mut response: impl FnMut(&mut T, usize, InferMetadata) -> InferResult,
) {
    match results {
        Ok((mut outputs, inference)) => {
            for (index, entry) in entries.into_iter().enumerate() {
                let metadata = InferMetadata {
                    memory: outputs.result_memory(),
                    prompt_tokens: entry.prompt_tokens,
                    tokenization: entry.tokenization,
                    queue: entry.queue_time.elapsed().saturating_sub(inference),
                    inference,
                };
                let _ = entry
                    .response_tx
                    .send(Ok(response(&mut outputs, index, metadata)));
            }
        }
        Err(error) => {
            for entry in entries {
                let _ = entry.response_tx.send(Err(error.clone()));
            }
        }
    }
}

#[derive(Debug)]
pub struct InferMetadata {
    pub memory: Option<ResultLease>,
    pub prompt_tokens: usize,
    pub tokenization: Duration,
    pub queue: Duration,
    pub inference: Duration,
}

#[derive(Debug)]
pub(crate) enum InferResult {
    Decision(DecisionInferResponse),
    Classification(ClassificationInferResponse),
    TokenClassification(TokenClassificationInferResponse),
    PooledEmbedding(PooledEmbeddingsInferResponse),
    AllEmbedding(AllEmbeddingsInferResponse),
}

impl InferResult {
    fn retain_for_request(&self) {
        let metadata = match self {
            Self::Decision(r) => &r.metadata,
            Self::Classification(r) => &r.metadata,
            Self::TokenClassification(r) => &r.metadata,
            Self::PooledEmbedding(r) => &r.metadata,
            Self::AllEmbedding(r) => &r.metadata,
        };
        retain_for_request(metadata.memory.clone());
    }
}

#[derive(Debug)]
pub struct DecisionInferResponse {
    pub results: infinity_v3_backend::DecisionOutput,
    pub metadata: InferMetadata,
}

#[derive(Debug)]
pub struct ClassificationInferResponse {
    pub results: Vec<f32>,
    pub metadata: InferMetadata,
}

fn make_token_prediction(
    index: usize,
    scores: Vec<f32>,
    input_ids: &[u32],
    tokens: &mut [String],
    offsets: &[(usize, usize)],
) -> TokenPrediction {
    let token_id = input_ids[index];
    let token = tokens
        .get_mut(index)
        .map(std::mem::take)
        .unwrap_or_else(|| format!("<token_{token_id}>"));
    let (start, end) = offsets
        .get(index)
        .map_or((None, None), |&(start, end)| (Some(start), Some(end)));
    (token, token_id, scores, start, end)
}

/// Token text, token ID, class scores, and optional character offsets.
pub type TokenPrediction = (String, u32, Vec<f32>, Option<usize>, Option<usize>);

#[derive(Debug)]
pub struct TokenClassificationInferResponse {
    pub results: Vec<TokenPrediction>,
    pub metadata: InferMetadata,
}

#[derive(Debug)]
pub struct PooledEmbeddingsInferResponse {
    pub results: Vec<f32>,
    pub metadata: InferMetadata,
}

#[derive(Debug)]
pub struct AllEmbeddingsInferResponse {
    pub results: Vec<Vec<f32>>,
    pub metadata: InferMetadata,
}

// Request rejections are expected; backend failures still need operator attention.
fn log_request_error(error: &TextEmbeddingsError) {
    if matches!(error, TextEmbeddingsError::Backend(backend) if !matches!(backend, BackendError::Overloaded(_)))
    {
        tracing::error!("{error}");
    } else {
        tracing::warn!("{error}");
    }
}

#[cfg(test)]
mod completion_tests {
    use super::*;

    #[tokio::test]
    async fn partial_client_batch_dispatches_before_next_admission() {
        let notify = Arc::new(Notify::new());
        let remaining = Arc::new(AtomicUsize::new(2));
        let admission = Arc::new(Semaphore::new(1));
        let queue = Queue::new(16, Some(1), 0.0, 1);
        let worker = async {
            for _ in 0..2 {
                notify.notified().await;
                let (entries, _) = queue.next_batch().await.unwrap();
                for entry in entries {
                    entry
                        .response_tx
                        .send(Err(BackendError::Inference("test reply".into())))
                        .unwrap();
                }
            }
        };
        let input = || {
            let queue = queue.clone();
            let notify = notify.clone();
            let remaining = remaining.clone();
            let admission = admission.clone();
            async move {
                let _permit = admission.acquire_owned().await.unwrap();
                let preparation = BatchPreparation {
                    remaining: Some(&remaining),
                    notify: &notify,
                };
                let (response_tx, response) = oneshot::channel();
                queue
                    .append(Entry {
                        encoding: crate::tokenization::ValidEncoding {
                            multimodal: None,
                            input_ids: vec![1],
                            token_type_ids: vec![0],
                            position_ids: vec![0],
                            tokens: vec![],
                            offsets: vec![],
                        },
                        metadata: Metadata {
                            input_memory: None,
                            decision: None,
                            client_batch: Some(remaining.clone()),
                            response_tx,
                            tokenization: Duration::ZERO,
                            queue_time: Instant::now(),
                            prompt_tokens: 1,
                            pooling: true,
                            token_classification: false,
                        },
                    })
                    .await;
                drop(preparation);
                assert!(response.await.unwrap().is_err());
            }
        };
        tokio::time::timeout(Duration::from_secs(2), async {
            tokio::join!(worker, input(), input());
        })
        .await
        .expect("admitted inputs must run before the full client batch is ready");
        assert_eq!(remaining.load(Ordering::SeqCst), 0);
        assert_eq!(admission.available_permits(), 1);
        queue.shutdown().await;
    }

    #[tokio::test]
    async fn failed_and_canceled_preparation_wakes_siblings() {
        let notify = Arc::new(Notify::new());
        let remaining = Arc::new(AtomicUsize::new(2));
        let fail = || -> Result<(), ()> {
            let _preparation = BatchPreparation {
                remaining: Some(&remaining),
                notify: &notify,
            };
            Err(())
        };
        assert!(fail().is_err());
        tokio::time::timeout(Duration::from_secs(2), notify.notified())
            .await
            .unwrap();
        assert_eq!(remaining.load(Ordering::SeqCst), 1);

        let worker_notify = notify.clone();
        let worker_remaining = remaining.clone();
        let (started, ready) = oneshot::channel();
        let task = tokio::spawn(async move {
            let _preparation = BatchPreparation {
                remaining: Some(&worker_remaining),
                notify: &worker_notify,
            };
            started.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        ready.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(2), notify.notified())
            .await
            .unwrap();
        assert_eq!(remaining.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn direct_token_response_owns_results_until_drop() {
        use infinity_v3_backend::result_memory::{ResultPool, REQUEST_RESULTS};
        assert!(REQUEST_RESULTS.try_with(|_| ()).is_err());
        let pool = ResultPool::new(192).unwrap();
        let deliver = || {
            let memory = pool.reserve(192)?;
            let mut outputs = infinity_v3_backend::TokenPredictions::default();
            outputs.memory = Some(memory);
            outputs.insert(0, vec![vec![0.25, 0.75]]);
            let (response_tx, receiver) = oneshot::channel();
            let entry = Metadata {
                input_memory: None,
                decision: None,
                client_batch: None,
                response_tx,
                tokenization: Duration::from_micros(7),
                queue_time: Instant::now(),
                prompt_tokens: 1,
                pooling: false,
                token_classification: true,
            };
            send_batch_results(
                vec![entry],
                Ok((outputs, Duration::from_millis(1))),
                |outputs, index, metadata| {
                    let scores = outputs.remove(&index).unwrap().remove(0);
                    InferResult::TokenClassification(TokenClassificationInferResponse {
                        results: vec![make_token_prediction(
                            0,
                            scores,
                            &[10],
                            &mut ["hello".into()],
                            &[(0, 5)],
                        )],
                        metadata,
                    })
                },
            );
            let InferResult::TokenClassification(response) =
                receiver.blocking_recv().unwrap().unwrap()
            else {
                panic!("unexpected response type");
            };
            Ok::<_, BackendError>(response)
        };
        let first = deliver().unwrap();
        assert_eq!(first.results[0].2, [0.25, 0.75]);
        assert_eq!(pool.used(), 192);
        assert!(matches!(deliver(), Err(BackendError::Overloaded(_))));
        drop(first);
        assert_eq!(pool.used(), 0);
        let recovered = deliver().unwrap();
        assert_eq!(recovered.results[0].0, "hello");
        assert_eq!(recovered.results[0].2, [0.25, 0.75]);
        drop(recovered);
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn cancellation_after_dispatch_preserves_response_order_and_timing() {
        let mut entries = Vec::new();
        let mut receivers = Vec::new();
        for prompt_tokens in [10, 20, 30] {
            let (response_tx, receiver) = oneshot::channel();
            receivers.push(receiver);
            entries.push(Metadata {
                input_memory: None,
                decision: None,
                client_batch: None,
                response_tx,
                tokenization: Duration::from_micros(7),
                queue_time: Instant::now(),
                prompt_tokens,
                pooling: false,
                token_classification: false,
            });
        }
        drop(receivers.remove(0));
        let inference = Duration::from_millis(1);
        let pool = infinity_v3_backend::result_memory::ResultPool::new(96).unwrap();
        let mut outputs = infinity_v3_backend::Predictions::default();
        outputs.memory = Some(pool.reserve(96).unwrap());
        for (index, value) in [1.0, 2.0, 3.0].into_iter().enumerate() {
            outputs.insert(index, vec![value]);
        }
        send_batch_results(
            entries,
            Ok((outputs, inference)),
            |outputs, index, metadata| {
                InferResult::Classification(ClassificationInferResponse {
                    results: outputs.remove(&index).unwrap(),
                    metadata,
                })
            },
        );
        assert_eq!(pool.used(), 96);
        assert!(pool.reserve(1).is_err());
        for (receiver, expected) in receivers.into_iter().zip([2.0, 3.0]) {
            let InferResult::Classification(response) = receiver.blocking_recv().unwrap().unwrap()
            else {
                panic!("unexpected response type");
            };
            assert_eq!(response.results, [expected]);
            assert_eq!(response.metadata.prompt_tokens, expected as usize * 10);
            assert_eq!(response.metadata.tokenization, Duration::from_micros(7));
            assert_eq!(response.metadata.inference, inference);
            assert_eq!(pool.used(), 96);
        }
        assert_eq!(pool.used(), 0);
        assert!(pool.reserve(96).is_ok());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_delivery_moves_text_and_preserves_alignment() {
        let ids = [10, 20];
        let mut tokens = vec!["hello".to_owned(), "東京".to_owned()];
        let offsets = [(0, 5), (6, 12)];
        let original = tokens[1].as_ptr();
        let result = make_token_prediction(1, vec![0.25, 0.75], &ids, &mut tokens, &offsets);
        assert_eq!(result.0.as_ptr(), original);
        assert_eq!(
            result,
            ("東京".into(), 20, vec![0.25, 0.75], Some(6), Some(12))
        );
        assert_eq!(tokens, ["hello", ""]);
        let first = make_token_prediction(0, vec![1.0], &ids, &mut tokens, &offsets);
        assert_eq!(first, ("hello".into(), 10, vec![1.0], Some(0), Some(5)));
        let fallback = make_token_prediction(1, vec![1.0], &ids, &mut [], &[]);
        assert_eq!(fallback, ("<token_20>".into(), 20, vec![1.0], None, None));
    }
}
