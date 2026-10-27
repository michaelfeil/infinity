//! Two-tier FIFO selection and packed-batch construction on a dedicated thread.
//!
//! `infer::batching_task` asks for batches; this worker owns the pending entries.
//! Prepared decision requests enter atomically. The preferred target cuts between
//! requests; only hard token/question limits may split one. Explicit targets stay
//! active under backlog.
//! Automatic multi-replica targets can relax under backlog. Selection never
//! runs a model; the selected entries' metadata follows the batch into delivery.

use crate::infer::InferResult;
use crate::log_response_dropped_after;
use crate::tokenization::ValidEncoding;
use infinity_v3_backend::{BackendError, Batch};
use std::cmp::max;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tracing::{instrument, Span};

fn early_dispatch_tokens(value: Option<&str>) -> Option<usize> {
    match value.map(str::trim) {
        None => Some(5_000),
        Some("") => None,
        Some(value) => {
            let tokens = value.parse::<usize>().unwrap_or_else(|_| {
                panic!(
                    "{} must be a nonnegative integer or empty",
                    infinity_v3_config::infinity_env!("EARLY_DISPATCH_TOKENS")
                )
            });
            (tokens != 0).then_some(tokens)
        }
    }
}

fn should_cut(current: usize, next: usize, target: Option<usize>) -> bool {
    current > 0 && target.is_some_and(|target| current + next > target)
}

fn target_for_backlog(
    queued_tokens: usize,
    target: Option<usize>,
    replicas: usize,
    fixed_target: bool,
) -> Option<usize> {
    target.filter(|_| fixed_target || queued_tokens < 20_000usize.saturating_mul(replicas.max(1)))
}

/// Queue entry
#[derive(Debug)]
pub struct Entry {
    /// Payload
    pub encoding: ValidEncoding,
    /// Entry metadata
    pub metadata: Metadata,
}

impl Entry {
    /// Charge a block prefix once per HTTP request in this batch, then every suffix.
    /// Exact host inputs must agree with native prefix sharing; other inputs pay full cost.
    fn batch_tokens(&self, selected: &[Self]) -> usize {
        use infinity_v3_backend::DecisionInput::BlockOptionTokens;
        let tokens = self.encoding.input_ids.len();
        let Some(BlockOptionTokens { prefix_length, .. }) = self.metadata.decision.as_ref() else {
            return tokens;
        };
        let prefix = *prefix_length;
        if prefix == 0 || prefix > tokens {
            return tokens;
        }
        let shared = selected.iter().any(|other| {
            matches!(other.metadata.decision.as_ref(), Some(BlockOptionTokens { prefix_length, .. }) if *prefix_length == prefix)
                && [
                    (&self.encoding.input_ids, &other.encoding.input_ids),
                    (&self.encoding.token_type_ids, &other.encoding.token_type_ids),
                    (&self.encoding.position_ids, &other.encoding.position_ids),
                ].into_iter().all(|(left, right)| {
                    left.get(..prefix).is_some_and(|left| right.get(..prefix) == Some(left))
                })
                && same_prefix_media(&self.encoding, &other.encoding, prefix)
        });
        if shared {
            tokens - prefix
        } else {
            tokens
        }
    }
}

/// Immutable image identity and all positional axes must agree within the prefix.
/// Distinct image allocations are conservatively charged even if pixels might match.
fn same_prefix_media(left: &ValidEncoding, right: &ValidEncoding, prefix: usize) -> bool {
    match (&left.multimodal, &right.multimodal) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            if !left.audios.is_empty()
                || !right.audios.is_empty()
                || left.position_ids.len() != right.position_ids.len()
                || !left
                    .position_ids
                    .iter()
                    .zip(&right.position_ids)
                    .all(|(left, right)| {
                        left.get(..prefix)
                            .is_some_and(|left| right.get(..prefix) == Some(left))
                    })
            {
                return false;
            }
            let mut left = left.images.iter().filter(|(start, _)| *start < prefix);
            let mut right = right.images.iter().filter(|(start, _)| *start < prefix);
            loop {
                match (left.next(), right.next()) {
                    (Some((a, x)), Some((b, y))) if a == b && Arc::ptr_eq(x, y) => {}
                    (None, None) => return true,
                    _ => return false,
                }
            }
        }
        _ => false,
    }
}

/// Entry metadata
#[derive(Debug)]
pub struct Metadata {
    pub(crate) priority: bool,
    /// Retained through queueing, model execution and result delivery.
    pub(crate) input_memory: Option<crate::input_memory::InputMemoryOwner>,
    pub decision: Option<infinity_v3_backend::DecisionInput>,
    /// InferResponse sender to communicate between the Infer struct and the batching_task
    pub(crate) response_tx: oneshot::Sender<Result<InferResult, BackendError>>,
    /// Tokenization duration
    pub(crate) tokenization: Duration,
    /// Instant when this entry was queued
    pub(crate) queue_time: Instant,
    /// Number of tokens in the prompt
    pub(crate) prompt_tokens: usize,
    /// Pooled embedding
    pub(crate) pooling: bool,
    pub(crate) token_classification: bool,
}

/// One prepared HTTP request. Questions stay together unless a hard limit splits it.
struct Request {
    entries: VecDeque<Entry>,
    priority: bool,
    shares_block_prefix: bool,
}

impl Request {
    fn new(entries: Vec<Entry>, shares_block_prefix: bool) -> Self {
        let priority = entries.first().is_some_and(|entry| entry.metadata.priority);
        Self {
            entries: entries.into(),
            priority,
            shares_block_prefix,
        }
    }

    fn tokens(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.encoding.input_ids.len())
            .sum()
    }
}

type PendingEntry = (Instant, Request);

// Monotonic actor admissions keep each tier FIFO; only the two heads compete.
struct PendingEntries {
    normal: VecDeque<PendingEntry>,
    priority: VecDeque<PendingEntry>,
    bonus: Duration,
}

impl PendingEntries {
    fn new(capacity: usize, bonus: Duration) -> Self {
        Self {
            normal: VecDeque::with_capacity(capacity),
            priority: VecDeque::new(),
            bonus,
        }
    }

    fn tier(&mut self, priority: bool) -> &mut VecDeque<PendingEntry> {
        if priority && !self.bonus.is_zero() {
            &mut self.priority
        } else {
            &mut self.normal
        }
    }

    fn push_back(&mut self, request: Request, now: Instant) {
        self.tier(request.priority).push_back((now, request));
    }

    fn push_front(&mut self, pending: PendingEntry) {
        self.tier(pending.1.priority).push_front(pending);
    }

    fn pop_front(&mut self) -> Option<PendingEntry> {
        let normal_first = match (self.normal.front(), self.priority.front()) {
            (Some((normal, _)), Some((priority, _))) => {
                priority.saturating_duration_since(*normal) >= self.bonus
            }
            (Some(_), None) => true,
            _ => false,
        };
        if normal_first {
            self.normal.pop_front()
        } else {
            self.priority.pop_front()
        }
    }

    fn drain(&mut self) -> impl Iterator<Item = Entry> + '_ {
        self.normal
            .drain(..)
            .chain(self.priority.drain(..))
            .flat_map(|(_, request)| request.entries)
    }

    fn len(&self) -> usize {
        self.normal
            .iter()
            .chain(&self.priority)
            .map(|(_, request)| request.entries.len())
            .sum()
    }
}

/// Request Queue
#[derive(Debug, Clone)]
pub struct Queue {
    worker: Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>,
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::Sender<QueueCommand>,
    closed: Arc<AtomicBool>,
}

impl Queue {
    pub fn new(
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        radix_mlp_threshold: f32,
        max_concurrent_requests: usize,
    ) -> Self {
        Self::with_replicas(
            max_batch_tokens,
            max_batch_requests,
            radix_mlp_threshold,
            max_concurrent_requests,
            1,
        )
    }

    pub fn with_replicas(
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        radix_mlp_threshold: f32,
        max_concurrent_requests: usize,
        replicas: usize,
    ) -> Self {
        Self::with_replicas_named(
            max_batch_tokens,
            max_batch_requests,
            radix_mlp_threshold,
            max_concurrent_requests,
            replicas,
            String::new(),
            Duration::from_secs(1),
        )
    }

    pub fn with_replicas_named(
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        radix_mlp_threshold: f32,
        max_concurrent_requests: usize,
        replicas: usize,
        model_name: String,
        priority_queue_bonus: Duration,
    ) -> Self {
        // Validate on the caller's thread before spawning the detached worker.
        let configured_target =
            match std::env::var(infinity_v3_config::infinity_env!("EARLY_DISPATCH_TOKENS")) {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(err) => panic!(
                    "Invalid {}: {err}",
                    infinity_v3_config::infinity_env!("EARLY_DISPATCH_TOKENS")
                ),
            };
        let fixed_target = configured_target.is_some();
        let early_dispatch_tokens = if replicas > 1 || fixed_target {
            early_dispatch_tokens(configured_target.as_deref())
        } else {
            None
        };
        // Create channels
        // https://github.com/huggingface/text-embeddings-inference/pull/726, b10 fix.
        let (queue_sender, queue_receiver) = mpsc::channel(2 * max_concurrent_requests);

        let closed = Arc::new(AtomicBool::new(false));
        let worker_closed = closed.clone();
        // Launch background queue task
        let worker = std::thread::spawn(move || {
            queue_blocking_task(
                max_batch_tokens,
                max_batch_requests,
                radix_mlp_threshold,
                max_concurrent_requests,
                queue_receiver,
                early_dispatch_tokens,
                fixed_target,
                replicas,
                worker_closed,
                model_name.into(),
                priority_queue_bonus,
            )
        });

        Self {
            worker: Arc::new(std::sync::Mutex::new(Some(worker))),
            queue_sender,
            closed,
        }
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    pub async fn shutdown(&self) {
        self.close();
        let handle = self.worker.lock().unwrap().take();
        if let Some(handle) = handle {
            let _ = tokio::task::spawn_blocking(move || handle.join()).await;
        }
    }

    /// Reject future appends and settle queued requests. Even if the channel is
    /// full, the worker observes the closed flag on its next command.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.queue_sender.try_send(QueueCommand::Close);
    }

    /// Append an independently prepared input, such as an embedding.
    pub async fn append(&self, entry: Entry) {
        self.append_request(vec![entry], false).await;
    }

    /// Admit a fully prepared HTTP request atomically, before the next batch can select it.
    #[instrument(skip_all)]
    pub async fn append_request(&self, mut entries: Vec<Entry>, shares_block_prefix: bool) {
        if entries.is_empty() {
            return;
        }
        for entry in &mut entries {
            if entry.metadata.input_memory.is_none() {
                entry.metadata.input_memory = crate::input_memory::current();
            }
        }
        if self.closed.load(Ordering::SeqCst) {
            for entry in entries {
                let _ = entry
                    .metadata
                    .response_tx
                    .send(Err(BackendError::Unhealthy));
            }
            return;
        }
        if let Err(err) = self
            .queue_sender
            .send(QueueCommand::Append(
                entries,
                shares_block_prefix,
                Span::current(),
            ))
            .await
        {
            if let QueueCommand::Append(entries, _, _) = err.0 {
                for entry in entries {
                    let _ = entry
                        .metadata
                        .response_tx
                        .send(Err(BackendError::Unhealthy));
                }
            }
        }
    }

    /// Get the next batch from the queue
    #[instrument(skip(self))]
    pub async fn next_batch(&self) -> Option<NextBatch> {
        let (response_sender, response_receiver) = oneshot::channel();

        // Closing a model can race with a pending batch request. Treat a closed
        // actor as an empty queue so the batching task can exit normally.
        if self.closed.load(Ordering::SeqCst) {
            return None;
        }
        if self
            .queue_sender
            .send(QueueCommand::NextBatch {
                response_sender,
                span: Span::current(),
            })
            .await
            .is_err()
        {
            self.close();
            return None;
        }
        match response_receiver.await {
            Ok(batch) => batch,
            Err(_) => {
                self.close();
                None
            }
        }
    }
}

// Background task responsible of the queue state
#[allow(clippy::too_many_arguments)]
fn queue_blocking_task(
    max_batch_tokens: usize,
    max_batch_requests: Option<usize>,
    radix_mlp_threshold: f32,
    max_concurrent_requests: usize,
    mut queue_receiver: mpsc::Receiver<QueueCommand>,
    early_dispatch_tokens: Option<usize>,
    fixed_target: bool,
    replicas: usize,
    closed: Arc<AtomicBool>,
    model_name: metrics::SharedString,
    priority_queue_bonus: Duration,
) {
    tracing::info!(
        ?early_dispatch_tokens,
        fixed_target,
        replicas,
        full_batch_backlog_tokens = 20_000usize.saturating_mul(replicas.max(1)),
        "Early dispatch token target"
    );
    let radix_mlp_pad: Option<usize> = std::env::var("RADIX_MLP_PAD")
        .ok()
        .and_then(|s| s.parse().ok());

    let mut entries = PendingEntries::new(max_concurrent_requests, priority_queue_bonus);
    // Canceled entries stay counted until removal. This may disable a soft
    // dispatch cut, but never bypasses the hard token/request limits.
    let mut queued_tokens = 0;
    // Reuse selection storage; outgoing batch vectors belong to the backend.
    let mut selected: Vec<Entry> = Vec::new();
    let allocate_token_buffers = || {
        (
            Vec::with_capacity(max_batch_tokens),
            Vec::with_capacity(max_batch_tokens),
            Vec::with_capacity(max_batch_tokens),
        )
    };
    // The first spare is ready before processing commands. Replacements are
    // staged on this thread after dispatch, while the backend owns the batch.
    let mut spare_token_buffers = allocate_token_buffers();

    while let Some(cmd) = queue_receiver.blocking_recv() {
        if closed.load(Ordering::SeqCst) {
            // Stop accepting commands, then settle those already buffered. The
            // worker must exit even while runtime owners still hold senders.
            queue_receiver.close();
            for entry in entries.drain() {
                let _ = entry
                    .metadata
                    .response_tx
                    .send(Err(BackendError::Unhealthy));
            }
            queued_tokens = 0;
            metrics::gauge!("infinity_queue_size", "model" => model_name.clone()).set(0.0);
            match cmd {
                QueueCommand::Append(entries, _, _) => {
                    for entry in entries {
                        let _ = entry
                            .metadata
                            .response_tx
                            .send(Err(BackendError::Unhealthy));
                    }
                }
                QueueCommand::NextBatch {
                    response_sender, ..
                } => {
                    let _ = response_sender.send(None);
                }
                QueueCommand::Close => {}
            }
            continue;
        }
        match cmd {
            QueueCommand::Close => {}
            QueueCommand::Append(request, shares_block_prefix, span) => {
                let _span = span.entered();
                let request = Request::new(request, shares_block_prefix);
                queued_tokens += request.tokens();
                let count = request.entries.len();
                // One admission time for every question in the HTTP request.
                entries.push_back(request, Instant::now());
                let gauge = metrics::gauge!("infinity_queue_size", "model" => model_name.clone());
                gauge.increment(count as f64);
            }
            QueueCommand::NextBatch {
                response_sender,
                span,
            } => {
                let _span = span.entered();
                // Snapshot the target once, including the batch being selected.
                // Draining a busy queue must not enable the cut midway through this batch.
                let target = target_for_backlog(
                    queued_tokens,
                    early_dispatch_tokens,
                    replicas,
                    fixed_target,
                );

                let mut current_tokens = 0;
                let mut pooled_count = 0;
                let mut token_count = 0;
                let mut offset_count = 0;
                while let Some((queued_at, mut request)) = entries.pop_front() {
                    request.entries.retain(|entry| {
                        if entry.metadata.response_tx.is_closed() {
                            queued_tokens -= entry.encoding.input_ids.len();
                            log_response_dropped_after("after tokenization / before batching");
                            false
                        } else {
                            true
                        }
                    });
                    if request.entries.is_empty() {
                        continue;
                    }
                    let mut request_tokens = 0;
                    let mut accepted = 0;
                    let mut request_selected = Vec::new();
                    // Cost only sharing inside this prepared request. A split starts
                    // with a fresh prefix charge on the next GPU batch.
                    while let Some(entry) = request.entries.pop_front() {
                        let work_tokens = if request.shares_block_prefix {
                            entry.batch_tokens(&request_selected)
                        } else {
                            entry.encoding.input_ids.len()
                        };
                        if current_tokens + request_tokens + work_tokens > max_batch_tokens
                            || max_batch_requests
                                .is_some_and(|max| selected.len() + accepted == max)
                        {
                            request.entries.push_front(entry);
                            break;
                        }
                        request_tokens += work_tokens;
                        accepted += 1;
                        request_selected.push(entry);
                    }
                    // Never split a request just to fill space left by another.
                    // An oversized request may be split only in an otherwise empty batch.
                    if !selected.is_empty()
                        && (!request.entries.is_empty()
                            || should_cut(current_tokens, request_tokens, target))
                    {
                        request_selected.extend(request.entries);
                        request.entries = request_selected.into();
                        entries.push_front((queued_at, request));
                        break;
                    }
                    if request_selected.is_empty() {
                        // A single prepared input exceeding the hard cap must fail,
                        // rather than remaining at the head of the queue forever.
                        if let Some(entry) = request.entries.pop_front() {
                            queued_tokens -= entry.encoding.input_ids.len();
                            let _ = entry.metadata.response_tx.send(Err(BackendError::Inference(
                                "Single input exceeds max batch tokens".into(),
                            )));
                        }
                        if !request.entries.is_empty() {
                            entries.push_front((queued_at, request));
                        }
                        continue;
                    }
                    for entry in request_selected {
                        queued_tokens -= entry.encoding.input_ids.len();
                        pooled_count += usize::from(entry.metadata.pooling);
                        token_count += entry.encoding.tokens.len();
                        offset_count += entry.encoding.offsets.len();
                        selected.push(entry);
                    }
                    current_tokens += request_tokens;
                    if !request.entries.is_empty() {
                        entries.push_front((queued_at, request));
                        break;
                    }
                    if Some(selected.len()) == max_batch_requests {
                        break;
                    }
                }

                // Packed batches use the staged numeric buffers. Other storage
                // follows the selected batch; token strings and offsets stay lazy.
                let capacity = selected.len();
                let (mut input_ids, mut token_type_ids, mut position_ids) = if capacity > 1 {
                    std::mem::take(&mut spare_token_buffers)
                } else {
                    (Vec::new(), Vec::new(), Vec::new())
                };
                let mut tokens = Vec::new();
                let mut offsets = Vec::new();
                let mut multimodal = Vec::with_capacity(capacity);
                let mut pooled_indices = Vec::with_capacity(pooled_count);
                let mut raw_indices = Vec::with_capacity(capacity - pooled_count);
                let mut metadata = Vec::with_capacity(capacity);
                let mut cu_seq_lengths =
                    Vec::with_capacity(if capacity == 0 { 0 } else { capacity + 1 });
                if capacity != 0 {
                    cu_seq_lengths.push(0);
                }
                let mut max_length = 0;

                for (entry_index, entry) in selected.drain(..).enumerate() {
                    let entry_tokens = entry.encoding.input_ids.len();
                    match entry.metadata.pooling {
                        true => pooled_indices.push(entry_index as u32),
                        false => raw_indices.push(entry_index as u32),
                    }
                    max_length = max(max_length, entry_tokens as u32);
                    multimodal.push(entry.encoding.multimodal);
                    if capacity == 1 {
                        // A single request can transfer its allocations directly.
                        input_ids = entry.encoding.input_ids;
                        token_type_ids = entry.encoding.token_type_ids;
                        position_ids = entry.encoding.position_ids;
                    } else {
                        input_ids.extend(entry.encoding.input_ids);
                        token_type_ids.extend(entry.encoding.token_type_ids);
                        position_ids.extend(entry.encoding.position_ids);
                    }
                    if entry_index == 0 {
                        tokens = entry.encoding.tokens;
                        offsets = entry.encoding.offsets;
                        tokens.reserve_exact(token_count - tokens.len());
                        offsets.reserve_exact(offset_count - offsets.len());
                    } else {
                        tokens.extend(entry.encoding.tokens);
                        offsets.extend(entry.encoding.offsets);
                    }
                    metadata.push(entry.metadata);
                    cu_seq_lengths.push(input_ids.len() as u32);
                }

                // Compute RadixMLP compact representation with BOTH mappings
                let (compact_input_ids, compact_position_ids, scatter_unfold, fold_gather) =
                    if radix_mlp_threshold > 1e-6 && !input_ids.is_empty() {
                        if let Some((compact_ids, compact_pos, scatter, fold)) =
                            crate::radix::compute(
                                &input_ids,
                                &position_ids,
                                &cu_seq_lengths,
                                &multimodal,
                                radix_mlp_pad,
                            )
                        {
                            // Only use if we achieved meaningful compression
                            let compression_ratio =
                                compact_ids.len() as f32 / input_ids.len() as f32;
                            tracing::info!(
                                "RadixMLP compression ratio: {:.2} ({} -> {})",
                                compression_ratio,
                                input_ids.len(),
                                compact_ids.len()
                            );
                            metrics::histogram!("infinity_radix_mlp_compression_ratio", "model" => model_name.clone())
                                .record(compression_ratio as f64);
                            if radix_mlp_threshold >= 1.0 || compression_ratio < radix_mlp_threshold
                            {
                                (
                                    Some(compact_ids),
                                    Some(compact_pos),
                                    Some(scatter),
                                    Some(fold),
                                )
                            } else {
                                (None, None, None, None)
                            }
                        } else {
                            (None, None, None, None)
                        }
                    } else {
                        (None, None, None, None)
                    };

                let batch_size = metadata.len();
                let next_batch = if metadata.is_empty() {
                    None
                } else {
                    Some((
                        metadata,
                        Batch {
                            multimodal,
                            input_ids,
                            token_type_ids,
                            position_ids,
                            cumulative_seq_lengths: cu_seq_lengths,
                            max_length,
                            pooled_indices,
                            raw_indices,
                            compact_input_ids,
                            compact_position_ids,
                            scatter_unfold,
                            fold_gather,
                            tokens,
                            offsets,
                        },
                    ))
                };

                if let Err(Some((metadata, _))) = response_sender.send(next_batch) {
                    for entry in metadata {
                        let _ = entry.response_tx.send(Err(BackendError::Inference(
                            "Batch pipeline exited before dispatch".into(),
                        )));
                    }
                }

                // Send first so the backend can start before allocating its
                // successor. Empty polls and single inputs retain the spare.
                if capacity > 1 {
                    spare_token_buffers = allocate_token_buffers();
                }

                let histogram =
                    metrics::histogram!("infinity_batch_next_size", "model" => model_name.clone());
                histogram.record(batch_size as f64);
                let histogram = metrics::histogram!("infinity_batch_next_tokens", "model" => model_name.clone());
                histogram.record(current_tokens as f64);
                let gauge = metrics::gauge!("infinity_queue_size", "model" => model_name.clone());
                gauge.set(entries.len() as f64)
            }
        }
    }
}

pub type NextBatch = (Vec<Metadata>, Batch);

#[derive(Debug)]
enum QueueCommand {
    Close,
    Append(Vec<Entry>, bool, Span),
    NextBatch {
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
}

pub(crate) fn prune_canceled_batch((metadata, batch): NextBatch) -> Option<NextBatch> {
    if metadata.iter().all(|m| !m.response_tx.is_closed()) {
        return Some((metadata, batch));
    }
    let mut kept = Vec::with_capacity(metadata.len());
    let mut packed = Batch {
        multimodal: vec![],
        input_ids: Vec::with_capacity(batch.input_ids.len()),
        token_type_ids: Vec::with_capacity(batch.token_type_ids.len()),
        position_ids: Vec::with_capacity(batch.position_ids.len()),
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: Vec::new(),
        raw_indices: Vec::new(),
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: Vec::new(),
    };
    for (index, entry) in metadata.into_iter().enumerate() {
        if entry.response_tx.is_closed() {
            log_response_dropped_after("after batching / before inference");
            continue;
        }
        if !batch.multimodal.is_empty() {
            packed.multimodal.push(batch.multimodal[index].clone());
        }
        let start = batch.cumulative_seq_lengths[index] as usize;
        let end = batch.cumulative_seq_lengths[index + 1] as usize;
        packed
            .input_ids
            .extend_from_slice(&batch.input_ids[start..end]);
        packed
            .token_type_ids
            .extend_from_slice(&batch.token_type_ids[start..end]);
        packed
            .position_ids
            .extend_from_slice(&batch.position_ids[start..end]);
        if !batch.tokens.is_empty() {
            packed.tokens.extend_from_slice(&batch.tokens[start..end]);
        }
        if !batch.offsets.is_empty() {
            packed.offsets.extend_from_slice(&batch.offsets[start..end]);
        }
        packed
            .cumulative_seq_lengths
            .push(packed.input_ids.len() as u32);
        packed.max_length = packed.max_length.max((end - start) as u32);
        if entry.pooling {
            packed.pooled_indices.push(kept.len() as u32);
        } else {
            packed.raw_indices.push(kept.len() as u32);
        }
        kept.push(entry);
    }
    if kept.is_empty() {
        None
    } else {
        Some((kept, packed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_queue(max_tokens: usize) -> Queue {
        test_queue_with_limits(max_tokens, None, None)
    }

    fn test_queue_with_limits(
        max_tokens: usize,
        max_requests: Option<usize>,
        target: Option<usize>,
    ) -> Queue {
        let (queue_sender, queue_receiver) = mpsc::channel(32);
        let closed = Arc::new(AtomicBool::new(false));
        let worker_closed = closed.clone();
        std::thread::spawn(move || {
            queue_blocking_task(
                max_tokens,
                max_requests,
                0.0,
                16,
                queue_receiver,
                target,
                true,
                1,
                worker_closed,
                "test".into(),
                Duration::from_secs(1),
            );
        });
        Queue {
            worker: Arc::new(std::sync::Mutex::new(None)),
            queue_sender,
            closed,
        }
    }

    fn test_entry(
        ids: Vec<u32>,
        pooling: bool,
    ) -> (Entry, oneshot::Receiver<Result<InferResult, BackendError>>) {
        let (response_tx, response_rx) = oneshot::channel();
        let len = ids.len();
        (
            Entry {
                encoding: ValidEncoding {
                    multimodal: None,
                    input_ids: ids,
                    token_type_ids: vec![0; len],
                    position_ids: (0..len as u32).collect(),
                    tokens: vec![],
                    offsets: vec![],
                },
                metadata: Metadata {
                    priority: false,
                    input_memory: None,
                    decision: None,
                    response_tx,
                    tokenization: Duration::ZERO,
                    queue_time: Instant::now(),
                    prompt_tokens: len,
                    pooling,
                    token_classification: false,
                },
            },
            response_rx,
        )
    }

    #[test]
    fn priority_gets_one_second_of_virtual_age_without_starving_older_work() {
        let mut queue = PendingEntries::new(8, Duration::from_secs(1));
        let base = Instant::now() - Duration::from_secs(10);
        let mut responses = Vec::new();
        // Equal virtual times retain append order, including across tiers.
        for (id, priority, millis) in [
            (1, false, 0),
            (2, false, 1_000),
            (3, false, 1_500),
            (4, true, 2_000),
            (5, true, 2_000),
        ] {
            let (mut entry, response) = test_entry(vec![id; 2], true);
            entry.metadata.priority = priority;
            entry.metadata.queue_time = base + Duration::from_millis(millis);
            responses.push(response);
            let now = base + Duration::from_millis(millis);
            queue.push_back(Request::new(vec![entry], false), now);
        }
        for id in [1, 2, 4, 5] {
            let (_, mut request) = queue.pop_front().unwrap();
            let entry = request.entries.pop_front().unwrap();
            assert_eq!(entry.encoding.input_ids, [id; 2]);
            if id == 4 {
                assert_eq!(entry.metadata.queue_time, base + Duration::from_secs(2));
            }
        }
        // Later priority arrivals cannot keep passing an older normal input.
        let (mut entry, response) = test_entry(vec![6; 2], true);
        entry.metadata.priority = true;
        entry.metadata.queue_time = base + Duration::from_millis(2_501);
        responses.push(response);
        queue.push_back(
            Request::new(vec![entry], false),
            base + Duration::from_millis(2_501),
        );
        for id in [3, 6] {
            let (_, mut request) = queue.pop_front().unwrap();
            let entry = request.entries.pop_front().unwrap();
            assert_eq!(entry.encoding.input_ids, [id; 2]);
        }
        assert!(queue.pop_front().is_none());
    }

    #[test]
    fn configured_bonus_and_zero_fifo_preserve_admission_order() {
        let base = Instant::now();
        for (bonus_ms, expected) in [(0, [1, 2, 3, 4]), (500, [1, 2, 3, 4]), (1000, [2, 1, 4, 3])] {
            let mut queue = PendingEntries::new(4, Duration::from_millis(bonus_ms));
            for (id, priority, millis) in [
                (1, false, 0),
                (2, true, 750),
                (3, false, 1000),
                (4, true, 1500),
            ] {
                let (mut entry, _) = test_entry(vec![id], true);
                entry.metadata.priority = priority;
                // Producer timestamps can arrive out of order; scheduling uses actor admission.
                entry.metadata.queue_time = base + Duration::from_secs(5 - u64::from(id));
                queue.push_back(
                    Request::new(vec![entry], false),
                    base + Duration::from_millis(millis),
                );
            }
            for id in expected {
                let pending = queue.pop_front().unwrap();
                assert_eq!(pending.1.entries.front().unwrap().encoding.input_ids, [id]);
                // A hard batch cut returns the same head without changing its scheduling time.
                queue.push_front(pending);
                assert_eq!(
                    queue
                        .pop_front()
                        .unwrap()
                        .1
                        .entries
                        .front()
                        .unwrap()
                        .encoding
                        .input_ids,
                    [id]
                );
            }
            assert!(queue.pop_front().is_none());
        }
    }

    #[tokio::test]
    async fn priority_entries_preserve_hard_cuts_and_cancellation() {
        let queue = test_queue(3);
        let mut responses = Vec::new();
        for id in [1, 2, 3] {
            let (mut entry, response) = test_entry(vec![id; 2], true);
            entry.metadata.priority = true;
            queue.append(entry).await;
            if id != 1 {
                responses.push(response);
            }
        }
        for id in [2, 3] {
            let (_, batch) = queue.next_batch().await.unwrap();
            assert_eq!(batch.input_ids, [id; 2]);
        }
        assert!(queue.next_batch().await.is_none());
        queue.shutdown().await;
    }

    #[tokio::test]
    async fn input_admission_survives_queue_and_batch_cancellation() {
        let queue = test_queue(16);
        let budget = Arc::new(tokio::sync::Semaphore::new(1));
        let (canceled, response) = test_entry(vec![1, 2], true);
        let owner = Arc::new(budget.clone().try_acquire_owned().unwrap());
        crate::input_memory::with_input_memory(owner, queue.append(canceled)).await;
        drop(response);
        assert_eq!(budget.available_permits(), 0);
        assert!(queue.next_batch().await.is_none());
        assert_eq!(budget.available_permits(), 1);

        let (live, _response) = test_entry(vec![3, 4], true);
        let owner = Arc::new(budget.clone().try_acquire_owned().unwrap());
        crate::input_memory::with_input_memory(owner, queue.append(live)).await;
        assert_eq!(budget.available_permits(), 0);
        let (metadata, batch) = queue.next_batch().await.unwrap();
        assert_eq!(batch.input_ids, [3, 4]);
        assert_eq!(budget.available_permits(), 0);
        drop(batch);
        // Completion metadata keeps admission until delivery or discard.
        assert_eq!(budget.available_permits(), 0);
        drop(metadata);
        assert_eq!(budget.available_permits(), 1);
    }

    #[tokio::test]
    async fn preferred_target_survives_backlog_and_cancellation() {
        let queue = test_queue_with_limits(40_960, Some(512), Some(16_384));
        let mut responses = Vec::new();
        let (canceled, response) = test_entry(vec![99; 16_384], true);
        drop(response);
        queue.append(canceled).await;
        // Six 16K HTTP requests remain separate even with roughly 100K queued.
        for id in 0..6 {
            let mut request = Vec::new();
            for question in 0..224 {
                let mut ids = vec![id; 2_048];
                ids.extend(vec![question + 100; 64]);
                let (mut entry, response) = test_entry(ids, true);
                entry.metadata.decision =
                    Some(infinity_v3_backend::DecisionInput::BlockOptionTokens {
                        prefix_length: 2_048,
                        token_ids: vec![1, 2],
                    });
                responses.push(response);
                request.push(entry);
            }
            queue.append_request(request, true).await;
        }
        for id in 0..6 {
            let (metadata, batch) = queue.next_batch().await.unwrap();
            assert_eq!(metadata.len(), 224);
            assert_eq!(batch.input_ids.len(), 224 * 2_112);
            assert!(batch.input_ids.chunks_exact(2_112).all(|ids| ids[0] == id));
        }
        assert!(queue.next_batch().await.is_none());
        queue.close();
    }

    #[tokio::test]
    async fn prefix_discount_requires_matching_attention_inputs() {
        for difference in 0..8 {
            let queue = test_queue_with_limits(8, None, None);
            let image = Arc::new(infinity_v3_backend::ImagePatches {
                pixels: vec![1.0],
                grid_thw: [1, 1, 1],
                patch_dim: 1,
                merge_size: 1,
            });
            let mut request = Vec::new();
            let mut responses = Vec::new();
            for question in 0..2 {
                let (mut entry, response) = test_entry(vec![1, 2, 3, 4, 5, 6], true);
                entry.metadata.decision =
                    Some(infinity_v3_backend::DecisionInput::BlockOptionTokens {
                        prefix_length: 4,
                        token_ids: vec![1, 2],
                    });
                if (4..=6).contains(&difference) {
                    let patches = if question == 1 && difference == 5 {
                        Arc::new(infinity_v3_backend::ImagePatches {
                            pixels: vec![2.0],
                            grid_thw: [1, 1, 1],
                            patch_dim: 1,
                            merge_size: 1,
                        })
                    } else {
                        image.clone()
                    };
                    let mut axes = [vec![0; 6], vec![0; 6], vec![0; 6]];
                    if question == 1 && difference == 6 {
                        axes[1][0] = 1;
                    }
                    entry.encoding.multimodal =
                        Some(Arc::new(infinity_v3_backend::MultimodalEncoding {
                            images: vec![(0, patches)],
                            audios: vec![],
                            position_ids: axes,
                            memory: None,
                        }));
                }
                if question == 1 {
                    match difference {
                        1 => entry.encoding.input_ids[0] = 9,
                        2 => entry.encoding.token_type_ids[0] = 1,
                        3 => entry.encoding.position_ids[0] = 1,
                        _ => {}
                    }
                }
                responses.push(response);
                request.push(entry);
            }
            queue.append_request(request, difference != 7).await;
            assert_eq!(
                queue.next_batch().await.unwrap().0.len(),
                if matches!(difference, 0 | 4) { 2 } else { 1 }
            );
            if !matches!(difference, 0 | 4) {
                assert_eq!(queue.next_batch().await.unwrap().0.len(), 1);
            }
            assert!(queue.next_batch().await.is_none());
            queue.close();
        }
    }

    #[tokio::test]
    async fn request_splits_respect_token_and_question_caps() {
        for (prefix, max_tokens, questions, expected) in
            [(16_384, 40_960, 512, 384), (2_048, 65_536, 513, 512)]
        {
            let queue = test_queue_with_limits(max_tokens, Some(512), Some(16_384));
            let mut request = Vec::new();
            let mut responses = Vec::new();
            for _ in 0..questions {
                let (mut entry, response) = test_entry(vec![1; prefix + 64], true);
                entry.metadata.decision =
                    Some(infinity_v3_backend::DecisionInput::BlockOptionTokens {
                        prefix_length: prefix,
                        token_ids: vec![1, 2],
                    });
                responses.push(response);
                request.push(entry);
            }
            queue.append_request(request, true).await;
            let (metadata, batch) = queue.next_batch().await.unwrap();
            assert_eq!(metadata.len(), expected);
            assert_eq!(batch.input_ids.len(), expected * (prefix + 64));
            assert_eq!(
                queue.next_batch().await.unwrap().0.len(),
                questions - expected
            );
            assert!(queue.next_batch().await.is_none());
            queue.close();
        }
    }

    #[tokio::test]
    async fn requests_stay_whole_across_soft_and_hard_cuts() {
        for target in [None, Some(3)] {
            let queue = test_queue_with_limits(9, Some(4), target);
            let mut responses = Vec::new();
            for ids in [vec![1], vec![2, 3, 4], vec![5]] {
                let mut request = Vec::new();
                for id in ids {
                    let (entry, response) = test_entry(vec![id; 3], true);
                    request.push(entry);
                    responses.push(response);
                }
                queue.append_request(request, true).await;
            }
            assert_eq!(queue.next_batch().await.unwrap().1.input_ids, [1; 3]);
            assert_eq!(
                queue.next_batch().await.unwrap().1.input_ids,
                [2, 2, 2, 3, 3, 3, 4, 4, 4]
            );
            assert_eq!(queue.next_batch().await.unwrap().1.input_ids, [5; 3]);
            assert!(queue.next_batch().await.is_none());
            queue.shutdown().await;
        }
    }

    #[tokio::test]
    async fn canceled_and_oversized_requests_do_not_block_later_work() {
        let queue = test_queue_with_limits(4, Some(2), Some(3));
        let budget = Arc::new(tokio::sync::Semaphore::new(1));
        let (canceled, response) = test_entry(vec![1; 4], true);
        let owner = Arc::new(budget.clone().try_acquire_owned().unwrap());
        crate::input_memory::with_input_memory(owner, queue.append_request(vec![canceled], false))
            .await;
        drop(response);
        let (oversized, failed) = test_entry(vec![2; 5], true);
        let (live, reply) = test_entry(vec![3; 4], true);
        queue.append_request(vec![oversized, live], false).await;
        let (mut metadata, batch) = queue.next_batch().await.unwrap();
        assert_eq!(batch.input_ids, [3; 4]);
        assert_eq!(budget.available_permits(), 1);
        assert!(matches!(
            failed.await.unwrap(),
            Err(BackendError::Inference(_))
        ));
        metadata
            .pop()
            .unwrap()
            .response_tx
            .send(Err(BackendError::Unhealthy))
            .unwrap();
        assert!(matches!(reply.await.unwrap(), Err(BackendError::Unhealthy)));
        assert!(queue.next_batch().await.is_none());
        queue.shutdown().await;
    }

    #[tokio::test]
    async fn staged_buffers_preserve_single_inputs_and_pending_batches() {
        let queue = test_queue(32);
        assert!(queue.next_batch().await.is_none());
        let (canceled, response) = test_entry(vec![0; 2], true);
        drop(response);
        queue.append(canceled).await;
        assert!(queue.next_batch().await.is_none());

        let (single, _response) = test_entry(vec![10, 11], true);
        let pointers = (
            single.encoding.input_ids.as_ptr(),
            single.encoding.token_type_ids.as_ptr(),
            single.encoding.position_ids.as_ptr(),
        );
        queue.append(single).await;
        let (_, single_batch) = queue.next_batch().await.unwrap();
        assert_eq!(
            pointers,
            (
                single_batch.input_ids.as_ptr(),
                single_batch.token_type_ids.as_ptr(),
                single_batch.position_ids.as_ptr(),
            )
        );
        assert_eq!(single_batch.tokens.capacity(), 0);
        assert_eq!(single_batch.offsets.capacity(), 0);

        // Keep previous batches alive while consuming and refilling the spare.
        let mut pending = Vec::new();
        for value in [20, 30, 40] {
            let (first, _first_response) = test_entry(vec![value; 2], true);
            let (mut second, _second_response) = test_entry(vec![value + 1; 3], false);
            second.metadata.token_classification = true;
            second.encoding.tokens = vec!["x".into(), "y".into(), "z".into()];
            second.encoding.offsets = vec![(0, 1), (2, 3), (4, 5)];
            queue.append(first).await;
            queue.append(second).await;
            let (metadata, batch) = queue.next_batch().await.unwrap();
            assert_eq!(metadata.len(), 2);
            assert!(metadata[1].token_classification);
            assert_eq!(
                batch.input_ids,
                [value, value, value + 1, value + 1, value + 1]
            );
            assert_eq!(batch.token_type_ids, [0; 5]);
            assert_eq!(batch.position_ids, [0, 1, 0, 1, 2]);
            assert_eq!(batch.cumulative_seq_lengths, [0, 2, 5]);
            assert_eq!(batch.max_length, 3);
            assert_eq!(batch.pooled_indices, [0]);
            assert_eq!(batch.raw_indices, [1]);
            assert_eq!(batch.tokens, ["x", "y", "z"]);
            assert_eq!(batch.offsets, [(0, 1), (2, 3), (4, 5)]);
            assert_eq!(batch.input_ids.capacity(), 32);
            assert_eq!(batch.token_type_ids.capacity(), 32);
            assert_eq!(batch.position_ids.capacity(), 32);
            assert_eq!(batch.tokens.capacity(), 3);
            assert_eq!(batch.offsets.capacity(), 3);
            pending.push(batch);
            assert!(queue.next_batch().await.is_none());
        }
        assert_eq!(single_batch.input_ids, [10, 11]);
        for (batch, value) in pending.iter().zip([20, 30, 40]) {
            assert_eq!(
                batch.input_ids,
                [value, value, value + 1, value + 1, value + 1]
            );
        }
    }

    #[test]
    fn cancellation_preserves_token_classification_offsets() {
        let (canceled, canceled_rx) = oneshot::channel();
        let (live, _live_rx) = oneshot::channel();
        drop(canceled_rx);
        let metadata = [canceled, live]
            .into_iter()
            .map(|response_tx| Metadata {
                priority: false,
                input_memory: None,
                response_tx,
                tokenization: Duration::ZERO,
                queue_time: Instant::now(),
                prompt_tokens: 1,
                pooling: false,
                token_classification: true,
                decision: None,
            })
            .collect();
        let permits = Arc::new(tokio::sync::Semaphore::new(2));
        let make_media = |value| {
            Arc::new(infinity_v3_backend::MultimodalEncoding {
                audios: vec![],
                images: vec![(
                    0,
                    Arc::new(infinity_v3_backend::ImagePatches {
                        pixels: vec![value],
                        grid_thw: [1, 1, 1],
                        patch_dim: 1,
                        merge_size: 1,
                    }),
                )],
                position_ids: [vec![0], vec![0], vec![0]],
                memory: Some(Arc::new(permits.clone().try_acquire_owned().unwrap())),
            })
        };
        let canceled_media = make_media(1.0);
        let live_media = make_media(2.0);
        let identity = Arc::downgrade(&live_media);
        let batch = Batch {
            multimodal: vec![Some(canceled_media), Some(live_media)],
            input_ids: vec![10, 20],
            token_type_ids: vec![0, 0],
            position_ids: vec![0, 0],
            cumulative_seq_lengths: vec![0, 1, 2],
            max_length: 1,
            pooled_indices: vec![],
            raw_indices: vec![0, 1],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec!["canceled".into(), "live".into()],
            offsets: vec![(0, 8), (2, 6)],
        };
        let (metadata, batch) = prune_canceled_batch((metadata, batch)).unwrap();
        assert_eq!(metadata.len(), 1);
        assert!(metadata[0].token_classification);
        assert_eq!(batch.input_ids, vec![20]);
        assert_eq!(batch.raw_indices, vec![0]);
        assert_eq!(batch.tokens, vec!["live"]);
        assert_eq!(batch.offsets, vec![(2, 6)]);
        assert_eq!(batch.multimodal.len(), 1);
        assert!(Arc::ptr_eq(
            batch.multimodal[0].as_ref().unwrap(),
            &identity.upgrade().unwrap()
        ));
        assert_eq!(
            batch.multimodal[0].as_ref().unwrap().images[0].1.pixels,
            vec![2.0]
        );
        assert_eq!(permits.available_permits(), 1);
        drop(batch);
        assert_eq!(permits.available_permits(), 2);
        assert_eq!(
            target_for_backlog(100_000, Some(5_000), 8, false),
            Some(5_000)
        );
        assert_eq!(target_for_backlog(160_000, Some(5_000), 8, false), None);
        assert_eq!(
            target_for_backlog(100_000, Some(16_384), 1, true),
            Some(16_384)
        );
    }
    #[test]
    fn cancellation_preserves_decision_metadata() {
        let (canceled, canceled_rx) = oneshot::channel();
        let (live, _live_rx) = oneshot::channel();
        drop(canceled_rx);
        let metadata = [canceled, live]
            .into_iter()
            .enumerate()
            .map(|(i, response_tx)| Metadata {
                priority: false,
                input_memory: None,
                response_tx,
                tokenization: Duration::ZERO,
                queue_time: Instant::now(),
                prompt_tokens: i + 2,
                pooling: false,
                token_classification: false,
                decision: Some(infinity_v3_backend::DecisionInput::Laya {
                    question_type: i * 2,
                    markers: if i == 0 { vec![0] } else { vec![1, 2] },
                }),
            })
            .collect();
        let batch = Batch {
            multimodal: vec![],
            input_ids: vec![10, 11, 20, 21, 22],
            token_type_ids: vec![0; 5],
            position_ids: vec![0, 1, 0, 1, 2],
            cumulative_seq_lengths: vec![0, 2, 5],
            max_length: 3,
            pooled_indices: vec![],
            raw_indices: vec![0, 1],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let (metadata, batch) = prune_canceled_batch((metadata, batch)).unwrap();
        assert_eq!(metadata.len(), 1);
        let infinity_v3_backend::DecisionInput::Laya {
            question_type,
            markers,
        } = metadata[0].decision.as_ref().unwrap()
        else {
            panic!("expected Laya metadata")
        };
        assert_eq!(*question_type, 2);
        assert_eq!(*markers, vec![1, 2]);
        assert_eq!(batch.input_ids, vec![20, 21, 22]);
        assert_eq!(batch.raw_indices, vec![0]);
        assert_eq!(batch.cumulative_seq_lengths, vec![0, 3]);
        assert_eq!(batch.position_ids, vec![0, 1, 2]);
    }
}
