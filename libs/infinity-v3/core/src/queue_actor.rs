//! Three-tier FIFO selection and packed-batch construction on a dedicated thread.
//!
//! `infer::batching_task` asks for batches; this worker owns the pending entries.
//! The early-dispatch target is a soft cut, relaxed for unfinished client batches
//! or a large backlog. Token and request maxima still apply. Selection never
//! runs a model; the selected entries' metadata follows the batch into delivery.

use crate::infer::InferResult;
use crate::log_response_dropped_after;
use crate::queue_budget::{QueueBudget, QueueLease, QueueTokenLimits};
use crate::tokenization::ValidEncoding;
use infinity_v3_backend::{BackendError, Batch};
use std::cmp::max;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
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
) -> Option<usize> {
    target.filter(|_| queued_tokens < 20_000usize.saturating_mul(replicas.max(1)))
}

/// Priority receives a virtual queue-age bonus; flex receives an offset. Equal virtual times retain admission order. The input alias "auto" selects default.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    #[default]
    #[serde(alias = "auto")]
    Default,
    Flex,
    Priority,
}

impl ServiceTier {
    pub fn is_priority(self) -> bool {
        self == Self::Priority
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Priority => "priority",
            Self::Flex => "flex",
        }
    }
}

/// Queue entry
#[derive(Debug)]
pub struct Entry {
    /// Payload
    pub encoding: ValidEncoding,
    /// Entry metadata
    pub metadata: Metadata,
}

/// Entry metadata
#[derive(Debug)]
pub struct Metadata {
    pub(crate) queue_reservation: Option<Arc<QueueLease>>,
    pub(crate) service_tier: ServiceTier,
    /// Retained through queueing, model execution and result delivery.
    pub(crate) input_memory: Option<crate::input_memory::InputMemoryOwner>,
    pub decision: Option<infinity_v3_backend::DecisionInput>,
    /// Shared HTTP-request identity and count of inputs still tokenizing.
    pub(crate) client_batch: Option<Arc<AtomicUsize>>,
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

impl Metadata {
    pub(crate) fn mark_dispatched(&self) {
        if let Some(reservation) = &self.queue_reservation {
            reservation.release();
        }
    }
}

struct PendingEntry {
    siblings: Vec<Entry>,
    queued_at: Instant,
    order: u128,
    entry: Entry,
}

// Each tier is FIFO. Only three heads compete, with fixed virtual enqueue times.
struct PendingEntries {
    normal: VecDeque<PendingEntry>,
    priority: VecDeque<PendingEntry>,
    flex: VecDeque<PendingEntry>,
    bonus: Duration,
    flex_offset: Duration,
    anchor: Instant,
    next_order: u128,
    len: usize,
}

impl PendingEntries {
    fn new(capacity: usize, bonus: Duration, flex_offset: Duration) -> Self {
        Self {
            normal: VecDeque::with_capacity(capacity),
            priority: VecDeque::new(),
            flex: VecDeque::new(),
            bonus,
            flex_offset,
            anchor: Instant::now(),
            next_order: 0,
            len: 0,
        }
    }

    fn tier(&mut self, tier: ServiceTier) -> &mut VecDeque<PendingEntry> {
        match tier {
            ServiceTier::Priority if !self.bonus.is_zero() => &mut self.priority,
            ServiceTier::Flex if !self.flex_offset.is_zero() => &mut self.flex,
            _ => &mut self.normal,
        }
    }

    fn push_back(&mut self, entry: Entry, now: Instant) {
        self.push_group(entry, Vec::new(), now);
    }

    fn push_group(&mut self, entry: Entry, siblings: Vec<Entry>, now: Instant) {
        self.len += 1 + siblings.len();
        if self.next_order == 0 {
            self.anchor = now;
        }
        let order = self.next_order;
        self.next_order = self.next_order.saturating_add(1);
        self.tier(entry.metadata.service_tier)
            .push_back(PendingEntry {
                siblings,
                queued_at: now,
                order,
                entry,
            });
    }

    fn push_front(&mut self, pending: PendingEntry) {
        self.len += 1 + pending.siblings.len();
        self.tier(pending.entry.metadata.service_tier)
            .push_front(pending);
    }

    fn pop_front(&mut self) -> Option<PendingEntry> {
        let tier = [
            (ServiceTier::Default, self.normal.front(), self.bonus),
            (ServiceTier::Priority, self.priority.front(), Duration::ZERO),
            (
                ServiceTier::Flex,
                self.flex.front(),
                self.bonus.saturating_add(self.flex_offset),
            ),
        ]
        .into_iter()
        .filter_map(|(tier, entry, offset)| {
            entry.map(|entry| {
                (
                    tier,
                    entry
                        .queued_at
                        .saturating_duration_since(self.anchor)
                        .saturating_add(offset),
                    entry.order,
                )
            })
        })
        .min_by_key(|(_, time, order)| (*time, *order))
        .map(|(tier, _, _)| tier)?;
        let pending = self.tier(tier).pop_front()?;
        self.len -= 1 + pending.siblings.len();
        Some(pending)
    }

    fn iter(&self) -> impl Iterator<Item = &Entry> {
        self.normal
            .iter()
            .chain(self.priority.iter())
            .chain(self.flex.iter())
            .flat_map(|pending| std::iter::once(&pending.entry).chain(&pending.siblings))
    }

    fn drain(&mut self) -> impl Iterator<Item = Entry> + '_ {
        self.len = 0;
        self.normal
            .drain(..)
            .chain(self.priority.drain(..))
            .chain(self.flex.drain(..))
            .flat_map(|pending| std::iter::once(pending.entry).chain(pending.siblings))
    }

    fn len(&self) -> usize {
        self.len
    }
}

fn unfinished_client(
    selected: &[Entry],
    next: &Entry,
    pending: &PendingEntries,
    clients: &mut Option<HashMap<usize, usize>>,
) -> bool {
    selected.iter().any(|entry| {
        entry.metadata.client_batch.as_ref().is_some_and(|client| {
            client.load(Ordering::SeqCst) != 0
                || next
                    .metadata
                    .client_batch
                    .as_ref()
                    .is_some_and(|next| Arc::ptr_eq(client, next))
                || clients
                    .get_or_insert_with(|| {
                        let mut counts = HashMap::new();
                        for entry in pending.iter() {
                            if let Some(client) = &entry.metadata.client_batch {
                                *counts.entry(Arc::as_ptr(client) as usize).or_default() += 1;
                            }
                        }
                        counts
                    })
                    .contains_key(&(Arc::as_ptr(client) as usize))
        })
    })
}

/// Request Queue
#[derive(Debug, Clone)]
pub struct Queue {
    group_limits: GroupLimits,
    budget: Arc<QueueBudget>,
    token_limits: Option<QueueTokenLimits>,
    priority_enabled: bool,
    flex_enabled: bool,
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
        Self::with_service_tiers_named(
            max_batch_tokens,
            max_batch_requests,
            radix_mlp_threshold,
            max_concurrent_requests,
            replicas,
            model_name,
            priority_queue_bonus,
            Duration::from_secs(10),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_service_tiers_named(
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        radix_mlp_threshold: f32,
        max_concurrent_requests: usize,
        replicas: usize,
        model_name: String,
        priority_queue_bonus: Duration,
        flex_queue_offset: Duration,
        max_batch_expanded_tokens: Option<usize>,
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
        let early_dispatch_tokens = if replicas > 1 || configured_target.is_some() {
            early_dispatch_tokens(configured_target.as_deref())
        } else {
            None
        };
        // Create channels
        // https://github.com/huggingface/text-embeddings-inference/pull/726, b10 fix.
        let (queue_sender, queue_receiver) = mpsc::channel(2 * max_concurrent_requests);

        let closed = Arc::new(AtomicBool::new(false));
        let worker_closed = closed.clone();
        let budget = Arc::new(QueueBudget::new(model_name.clone()));
        // Launch background queue task
        let worker = std::thread::spawn(move || {
            queue_blocking_task(
                max_batch_tokens,
                max_batch_requests,
                radix_mlp_threshold,
                max_concurrent_requests,
                queue_receiver,
                early_dispatch_tokens,
                replicas,
                worker_closed,
                model_name.into(),
                priority_queue_bonus,
                flex_queue_offset,
                max_batch_expanded_tokens.unwrap_or(max_batch_tokens),
            )
        });

        Self {
            group_limits: GroupLimits {
                compute: max_batch_tokens,
                expanded: max_batch_expanded_tokens.unwrap_or(max_batch_tokens),
                requests: max_batch_requests,
                prefix_aware: max_batch_expanded_tokens.unwrap_or(max_batch_tokens)
                    > max_batch_tokens
                    && radix_mlp_threshold == 0.0,
            },
            budget,
            token_limits: None,
            priority_enabled: !priority_queue_bonus.is_zero(),
            flex_enabled: !flex_queue_offset.is_zero(),
            worker: Arc::new(std::sync::Mutex::new(Some(worker))),
            queue_sender,
            closed,
        }
    }

    pub(crate) fn flex_enabled(&self) -> bool {
        self.flex_enabled
    }

    pub(crate) fn priority_enabled(&self) -> bool {
        self.priority_enabled
    }

    /// Limits are shared by all tiers and all replicas consuming this queue.
    pub fn with_token_limits(mut self, limits: Option<QueueTokenLimits>) -> Self {
        self.token_limits = limits;
        if limits.is_some() {
            self.priority_enabled = true;
            self.flex_enabled = true;
        }
        self
    }

    pub fn queued_compute_tokens(&self) -> i64 {
        self.budget.queued_tokens()
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

    /// Append an entry to the queue
    #[instrument(skip_all)]
    pub async fn append(&self, mut entry: Entry) -> Result<QueueLease, crate::TextEmbeddingsError> {
        let lease = self.budget.reserve(
            i64::try_from(entry.encoding.input_ids.len()).map_err(|_| {
                crate::TextEmbeddingsError::Validation(
                    "Queue token charge exceeds supported range".into(),
                )
            })?,
            entry.metadata.service_tier,
            self.token_limits,
        )?;
        entry.metadata.queue_reservation = Some(lease.queue_guard());
        if entry.metadata.input_memory.is_none() {
            entry.metadata.input_memory = crate::input_memory::current();
        }
        if self.closed.load(Ordering::SeqCst) {
            let _ = entry
                .metadata
                .response_tx
                .send(Err(BackendError::Unhealthy));
            return Err(BackendError::Unhealthy.into());
        }
        if let Err(err) = self
            .queue_sender
            .send(QueueCommand::Append(Box::new(entry), Span::current()))
            .await
        {
            if let QueueCommand::Append(entry, _) = err.0 {
                let _ = entry
                    .metadata
                    .response_tx
                    .send(Err(BackendError::Unhealthy));
            }
            return Err(BackendError::Unhealthy.into());
        }
        Ok(lease)
    }

    /// Enqueue an already-prepared block request atomically. Never partially
    /// execute a request whose deduplicated billing assumes one prefix prefill.
    pub async fn append_group(
        &self,
        mut group: Vec<Entry>,
    ) -> Result<QueueLease, crate::TextEmbeddingsError> {
        let (compute, expanded) = self.group_limits.charge(&group);
        if compute > self.group_limits.compute
            || expanded > self.group_limits.expanded
            || self
                .group_limits
                .requests
                .is_some_and(|limit| group.len() > limit)
        {
            return Err(crate::TextEmbeddingsError::Validation(format!(
                "Block-attention request must fit one batch: {compute} compute tokens, {expanded} expanded tokens, {} questions; limits are {}, {}, {:?}",
                group.len(), self.group_limits.compute, self.group_limits.expanded, self.group_limits.requests
            )));
        }
        let tier = group
            .first()
            .map_or(ServiceTier::Default, |entry| entry.metadata.service_tier);
        let lease = self.budget.reserve(
            i64::try_from(compute).map_err(|_| {
                crate::TextEmbeddingsError::Validation(
                    "Queue token charge exceeds supported range".into(),
                )
            })?,
            tier,
            self.token_limits,
        )?;
        let queued_lease = lease.queue_guard();
        for entry in &mut group {
            entry.metadata.queue_reservation = Some(queued_lease.clone());
            entry.metadata.input_memory = crate::input_memory::current();
        }
        if self.is_closed() {
            return Err(BackendError::Unhealthy.into());
        }
        self.queue_sender
            .send(QueueCommand::AppendGroup(group, Span::current()))
            .await
            .map_err(|_| crate::TextEmbeddingsError::Backend(BackendError::Unhealthy))?;
        Ok(lease)
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

#[derive(Debug, Clone, Copy)]
struct GroupLimits {
    compute: usize,
    expanded: usize,
    requests: Option<usize>,
    prefix_aware: bool,
}

impl GroupLimits {
    fn charge(&self, entries: &[Entry]) -> (usize, usize) {
        let mut prefixes = HashSet::new();
        let mut compute = 0usize;
        let mut expanded = 0usize;
        for entry in entries {
            let len = entry.encoding.input_ids.len();
            let prefix = self
                .prefix_aware
                .then(|| BlockPrefix::from_entry(entry))
                .flatten();
            let saved = prefix.map_or(0, |prefix| {
                let length = prefix.ids.len();
                if prefixes.insert(prefix) {
                    0
                } else {
                    length
                }
            });
            compute = compute.saturating_add(len - saved);
            expanded = expanded.saturating_add(len);
        }
        (compute, expanded)
    }
}

/// Conservative subset of the native `same_block_prefix` identity. Images
/// share only when they reuse the same prepared patches; equal independent
/// allocations are charged in full. The group retains those Arcs during
/// accounting, so pointer reuse cannot create a match.
#[derive(PartialEq, Eq, Hash)]
struct BlockPrefix<'a> {
    ids: &'a [u32],
    types: &'a [u32],
    positions: &'a [u32],
    axes: Option<[&'a [u32]; 3]>,
    images: Vec<(usize, usize)>,
}

impl<'a> BlockPrefix<'a> {
    fn from_entry(entry: &'a Entry) -> Option<Self> {
        let infinity_v3_backend::DecisionInput::BlockOptionTokens { prefix_length, .. } =
            entry.metadata.decision.as_ref()?
        else {
            return None;
        };
        let len = *prefix_length;
        if len == 0 {
            return None;
        }
        let encoding = &entry.encoding;
        let mut prefix = Self {
            ids: encoding.input_ids.get(..len)?,
            types: encoding.token_type_ids.get(..len)?,
            positions: encoding.position_ids.get(..len)?,
            axes: None,
            images: Vec::new(),
        };
        if let Some(media) = &encoding.multimodal {
            // Native block attention rejects audio and images crossing boundaries.
            // Be conservative for suffix images as well; no discount is necessary.
            if !media.audios.is_empty() || media.radix_prefix_len()? > len {
                return None;
            }
            let [a, b, c] = &media.position_ids;
            prefix.axes = Some([a.get(..len)?, b.get(..len)?, c.get(..len)?]);
            prefix.images = media
                .images
                .iter()
                .map(|(start, patches)| (*start, Arc::as_ptr(patches) as usize))
                .collect();
        }
        Some(prefix)
    }
}

// Background task responsible for the queue state.
#[allow(clippy::too_many_arguments)]
fn queue_blocking_task(
    max_batch_tokens: usize,
    max_batch_requests: Option<usize>,
    radix_mlp_threshold: f32,
    max_concurrent_requests: usize,
    mut queue_receiver: mpsc::Receiver<QueueCommand>,
    early_dispatch_tokens: Option<usize>,
    replicas: usize,
    closed: Arc<AtomicBool>,
    model_name: metrics::SharedString,
    priority_queue_bonus: Duration,
    flex_queue_offset: Duration,
    max_batch_expanded_tokens: usize,
) {
    tracing::info!(
        ?early_dispatch_tokens,
        max_batch_tokens,
        max_batch_expanded_tokens,
        replicas,
        full_batch_backlog_tokens = 20_000usize.saturating_mul(replicas.max(1)),
        "Early dispatch token target"
    );
    let radix_mlp_pad: Option<usize> = std::env::var("RADIX_MLP_PAD")
        .ok()
        .and_then(|s| s.parse().ok());

    let mut entries = PendingEntries::new(
        max_concurrent_requests,
        priority_queue_bonus,
        flex_queue_offset,
    );
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
                QueueCommand::AppendGroup(group, _) => {
                    for entry in group {
                        let _ = entry
                            .metadata
                            .response_tx
                            .send(Err(BackendError::Unhealthy));
                    }
                }
                QueueCommand::Append(entry, _) => {
                    let _ = entry
                        .metadata
                        .response_tx
                        .send(Err(BackendError::Unhealthy));
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
            QueueCommand::AppendGroup(group, span) => {
                let _span = span.entered();
                queued_tokens += group
                    .iter()
                    .map(|e| e.encoding.input_ids.len())
                    .sum::<usize>();
                metrics::gauge!("infinity_queue_size", "model" => model_name.clone())
                    .increment(group.len() as f64);
                let mut group = group.into_iter();
                if let Some(first) = group.next() {
                    entries.push_group(first, group.collect(), Instant::now());
                }
            }
            QueueCommand::Append(entry, span) => {
                let _span = span.entered();
                queued_tokens += entry.encoding.input_ids.len();
                // Actor admission is monotonic even when senders race or wait on channel capacity.
                entries.push_back(*entry, Instant::now());
                let gauge = metrics::gauge!("infinity_queue_size", "model" => model_name.clone());
                gauge.increment(1.0);
            }
            QueueCommand::NextBatch {
                response_sender,
                span,
            } => {
                let _span = span.entered();
                // Snapshot the target once, including the batch being selected.
                // Draining a busy queue must not enable the cut midway through this batch.
                let target = target_for_backlog(queued_tokens, early_dispatch_tokens, replicas);

                let mut current_tokens = 0usize;
                let mut expanded_tokens = 0usize;
                let mut pooled_count = 0;
                let mut token_count = 0;
                let mut offset_count = 0;
                // Build lookahead only for a soft cut, then update it on removal.
                // Consecutive inputs from the same HTTP request avoid this map.
                // Canceled tails stay counted until popped; only soft cuts can change.
                let mut pending_clients: Option<HashMap<usize, usize>> = None;

                while let Some(pending) = entries.pop_front() {
                    if !pending.siblings.is_empty() {
                        // A request is indivisible, but several whole requests
                        // can share one batch. Charge each request independently;
                        // extra native sharing across requests is only a bonus.
                        let queued_at = pending.queued_at;
                        let order = pending.order;
                        let mut group: Vec<_> = std::iter::once(pending.entry)
                            .chain(pending.siblings)
                            .collect();
                        queued_tokens -= group
                            .iter()
                            .map(|e| e.encoding.input_ids.len())
                            .sum::<usize>();
                        group.retain(|entry| !entry.metadata.response_tx.is_closed());
                        // Filtering a group can remove queued client entries;
                        // rebuild soft-cut lookahead from the remaining queue.
                        pending_clients = None;
                        let Some(first) = group.first() else {
                            continue;
                        };
                        let limits = GroupLimits {
                            compute: max_batch_tokens,
                            expanded: max_batch_expanded_tokens,
                            requests: max_batch_requests,
                            prefix_aware: max_batch_expanded_tokens > max_batch_tokens
                                && radix_mlp_threshold == 0.0,
                        };
                        let (compute, expanded) = limits.charge(&group);
                        let soft_cut = should_cut(current_tokens, compute, target)
                            && !unfinished_client(&selected, first, &entries, &mut pending_clients);
                        if soft_cut
                            || current_tokens.saturating_add(compute) > max_batch_tokens
                            || expanded_tokens.saturating_add(expanded) > max_batch_expanded_tokens
                            || max_batch_requests
                                .is_some_and(|max| selected.len() + group.len() > max)
                        {
                            queued_tokens += expanded;
                            let mut group = group.into_iter();
                            if let Some(entry) = group.next() {
                                entries.push_front(PendingEntry {
                                    siblings: group.collect(),
                                    queued_at,
                                    order,
                                    entry,
                                });
                            }
                            break;
                        }
                        current_tokens += compute;
                        expanded_tokens += expanded;
                        for entry in group {
                            pooled_count += usize::from(entry.metadata.pooling);
                            token_count += entry.encoding.tokens.len();
                            offset_count += entry.encoding.offsets.len();
                            selected.push(entry);
                        }
                        if Some(selected.len()) == max_batch_requests {
                            break;
                        }
                        continue;
                    }
                    let PendingEntry {
                        queued_at,
                        order,
                        entry,
                        ..
                    } = pending;
                    if let (Some(clients), Some(client)) =
                        (&mut pending_clients, &entry.metadata.client_batch)
                    {
                        let id = Arc::as_ptr(client) as usize;
                        let count = clients.get_mut(&id).expect("queued client counted");
                        *count -= 1;
                        if *count == 0 {
                            clients.remove(&id);
                        }
                    }
                    // Filter entries where the response receiver was dropped (== entries where the request
                    // was dropped by the client)
                    if entry.metadata.response_tx.is_closed() {
                        queued_tokens -= entry.encoding.input_ids.len();
                        log_response_dropped_after("after tokenization / before batching");
                        continue;
                    }

                    let entry_tokens = entry.encoding.input_ids.len();
                    // Soft limit: always allow one input, even if it exceeds the target.
                    if should_cut(current_tokens, entry_tokens, target)
                        && !unfinished_client(&selected, &entry, &entries, &mut pending_clients)
                    {
                        entries.push_front(PendingEntry {
                            siblings: Vec::new(),
                            queued_at,
                            order,
                            entry,
                        });
                        break;
                    }

                    let total_tokens = current_tokens + entry_tokens;

                    if total_tokens > max_batch_tokens
                        || expanded_tokens.saturating_add(entry_tokens) > max_batch_expanded_tokens
                    {
                        debug_assert!(
                            entry_tokens <= max_batch_tokens,
                            "Single entry exceeds max batch tokens, leading to stall."
                        );
                        entries.push_front(PendingEntry {
                            siblings: Vec::new(),
                            queued_at,
                            order,
                            entry,
                        });
                        break;
                    }

                    queued_tokens -= entry_tokens;
                    current_tokens = total_tokens;
                    expanded_tokens += entry_tokens;
                    pooled_count += usize::from(entry.metadata.pooling);
                    token_count += entry.encoding.tokens.len();
                    offset_count += entry.encoding.offsets.len();
                    selected.push(entry);
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
                histogram.record(expanded_tokens as f64);
                metrics::histogram!("infinity_batch_next_compute_tokens", "model" => model_name.clone())
                    .record(current_tokens as f64);
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
    Append(Box<Entry>, Span),
    AppendGroup(Vec<Entry>, Span),
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
        test_queue_with_expanded_limit(max_tokens, max_requests, target, max_tokens)
    }

    fn test_queue_with_expanded_limit(
        max_tokens: usize,
        max_requests: Option<usize>,
        target: Option<usize>,
        max_expanded_tokens: usize,
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
                1,
                worker_closed,
                "test".into(),
                Duration::from_secs(1),
                Duration::from_secs(1),
                max_expanded_tokens,
            );
        });
        Queue {
            group_limits: GroupLimits {
                compute: max_tokens,
                expanded: max_expanded_tokens,
                requests: max_requests,
                prefix_aware: max_expanded_tokens > max_tokens,
            },
            budget: Arc::new(QueueBudget::default()),
            token_limits: None,
            priority_enabled: true,
            flex_enabled: true,
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
                    service_tier: ServiceTier::Default,
                    input_memory: None,
                    queue_reservation: None,
                    decision: None,
                    client_batch: None,
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

    fn block_entry(
        prefix: usize,
        suffix: usize,
    ) -> (Entry, oneshot::Receiver<Result<InferResult, BackendError>>) {
        let (mut entry, response) = test_entry(vec![1; prefix + suffix], false);
        entry.metadata.decision = Some(infinity_v3_backend::DecisionInput::BlockOptionTokens {
            token_ids: vec![7, 8],
            prefix_length: prefix,
        });
        (entry, response)
    }

    #[tokio::test]
    async fn thirteen_k_state_and_fifty_questions_are_atomic_or_rejected() {
        for expanded_limit in [524_288, 700_000] {
            let queue = test_queue_with_expanded_limit(32_768, None, Some(5_000), expanded_limit);
            let mut responses = Vec::new();
            let mut group = Vec::new();
            for question in 0..50 {
                let (mut entry, response) = block_entry(13_000, 150);
                entry.encoding.input_ids[13_000..].fill(100 + question);
                responses.push(response);
                group.push(entry);
            }
            if expanded_limit == 524_288 {
                assert!(matches!(
                    queue.append_group(group).await,
                    Err(crate::TextEmbeddingsError::Validation(_))
                ));
                assert!(queue.next_batch().await.is_none());
            } else {
                // Work already queued must not consume part of this request's budget.
                let (earlier, response) = block_entry(20_000, 100);
                responses.push(response);
                let _earlier_lease = queue.append(earlier).await.unwrap();
                let _group_lease = queue.append_group(group).await.unwrap();
                assert_eq!(queue.next_batch().await.unwrap().0.len(), 1);
                let (metadata, batch) = queue.next_batch().await.unwrap();
                assert_eq!(metadata.len(), 50);
                assert_eq!(batch.input_ids.len(), 657_500);
                for (question, ids) in batch.input_ids.chunks_exact(13_150).enumerate() {
                    assert!(ids[13_000..].iter().all(|id| *id == 100 + question as u32));
                }
                assert!(queue.next_batch().await.is_none());
            }
            queue.shutdown().await;
        }
    }

    #[tokio::test]
    async fn whole_requests_share_batches_without_cross_request_discounts_or_splits() {
        // Each request costs 6 prefix + 2*2 suffix = 10 compute / 16 expanded.
        // Identical prefixes in different requests still receive separate charges.
        for (compute, expanded, requests, target, expected) in [
            (20, 40, None, None, vec![4]),
            (19, 40, None, None, vec![2, 2]),
            (20, 31, None, None, vec![2, 2]),
            (20, 40, Some(3), None, vec![2, 2]),
            (20, 40, None, Some(15), vec![2, 2]),
        ] {
            let queue = test_queue_with_expanded_limit(compute, requests, target, expanded);
            let mut responses = Vec::new();
            for request in 0..2 {
                let mut group = Vec::new();
                for question in 0..2 {
                    let (mut entry, response) = block_entry(6, 2);
                    entry.encoding.input_ids[6..].fill(10 + request * 2 + question);
                    group.push(entry);
                    responses.push(response);
                }
                queue.append_group(group).await.unwrap();
            }
            let mut observed = Vec::new();
            for count in expected {
                let (metadata, batch) = queue.next_batch().await.unwrap();
                assert_eq!(metadata.len(), count);
                observed.extend(batch.input_ids.chunks_exact(8).map(|row| row[7]));
            }
            assert_eq!(observed, [10, 11, 12, 13]);
            assert!(queue.next_batch().await.is_none());
            queue.shutdown().await;
        }
    }

    #[tokio::test]
    async fn regular_rows_respect_expanded_limit_around_atomic_groups() {
        for group_first in [false, true] {
            for expanded in [23, 24] {
                let queue = test_queue_with_expanded_limit(20, None, None, expanded);
                let (single, _single_rx) = block_entry(6, 2);
                let (a, _a_rx) = block_entry(6, 2);
                let (b, _b_rx) = block_entry(6, 2);
                if group_first {
                    queue.append_group(vec![a, b]).await.unwrap();
                    queue.append(single).await;
                } else {
                    queue.append(single).await;
                    queue.append_group(vec![a, b]).await.unwrap();
                }
                if expanded == 24 {
                    assert_eq!(queue.next_batch().await.unwrap().0.len(), 3);
                } else {
                    let first = if group_first { 2 } else { 1 };
                    assert_eq!(queue.next_batch().await.unwrap().0.len(), first);
                    assert_eq!(queue.next_batch().await.unwrap().0.len(), 3 - first);
                }
                assert!(queue.next_batch().await.is_none());
                queue.shutdown().await;
            }
        }
    }

    #[tokio::test]
    async fn atomic_groups_reject_compute_and_request_overflow_and_release_cancellation() {
        for (compute, requests) in [(9, None), (10, Some(1))] {
            let queue = test_queue_with_expanded_limit(compute, requests, None, 100);
            let (a, _a_response) = block_entry(6, 2);
            let (b, _b_response) = block_entry(6, 2);
            assert!(queue.append_group(vec![a, b]).await.is_err());
            assert!(queue.next_batch().await.is_none());
            queue.shutdown().await;
        }
        let queue = test_queue_with_expanded_limit(10, None, None, 100);
        let (a, a_response) = block_entry(6, 2);
        let (b, b_response) = block_entry(6, 2);
        let _group_lease = queue.append_group(vec![a, b]).await.unwrap();
        drop((a_response, b_response));
        assert!(queue.next_batch().await.is_none());
        let (live, _response) = block_entry(6, 2);
        let _live_lease = queue.append(live).await.unwrap();
        assert_eq!(queue.next_batch().await.unwrap().0.len(), 1);
        queue.shutdown().await;
    }

    #[tokio::test]
    async fn block_groups_do_not_discount_unrelated_prefixes() {
        for difference in 0..5 {
            let queue = test_queue_with_expanded_limit(10, None, None, 100);
            let (first, first_response) = block_entry(6, 2);
            let (mut second, second_response) = block_entry(6, 2);
            match difference {
                0 => second.encoding.input_ids[0] = 2,
                1 => second.encoding.token_type_ids[0] = 1,
                2 => second.encoding.position_ids[0] = 1,
                3 => {
                    second.metadata.decision =
                        Some(infinity_v3_backend::DecisionInput::OptionTokens {
                            token_ids: vec![7, 8],
                        })
                }
                _ => {
                    second.metadata.decision =
                        Some(infinity_v3_backend::DecisionInput::BlockOptionTokens {
                            token_ids: vec![7, 8],
                            prefix_length: 5,
                        })
                }
            }
            assert!(queue.append_group(vec![first, second]).await.is_err());
            assert!(queue.next_batch().await.is_none());
            drop((first_response, second_response));
            queue.shutdown().await;
        }
    }

    #[tokio::test]
    async fn block_budget_requires_shared_image_patches_and_matching_positions() {
        use infinity_v3_backend::{ImagePatches, MultimodalEncoding};
        let patches = Arc::new(ImagePatches {
            pixels: vec![1.0],
            grid_thw: [1, 1, 1],
            patch_dim: 1,
            merge_size: 1,
        });
        for difference in 0..4 {
            let queue = test_queue_with_expanded_limit(10, None, None, 100);
            let mut responses = Vec::new();
            let mut group = Vec::new();
            for row in 0..2 {
                let (mut entry, response) = block_entry(6, 2);
                let mut media = MultimodalEncoding {
                    images: vec![(1, patches.clone())],
                    audios: vec![],
                    position_ids: std::array::from_fn(|_| (0..8).collect()),
                    memory: None,
                };
                if row == 1 {
                    match difference {
                        1 => {
                            media.images[0].1 = Arc::new(ImagePatches {
                                pixels: vec![2.0],
                                ..(*patches).clone()
                            })
                        }
                        2 => media.position_ids[1][5] = 99,
                        3 => media.images[0].0 = 2,
                        _ => {}
                    }
                }
                entry.encoding.multimodal = Some(Arc::new(media));
                responses.push(response);
                group.push(entry);
            }
            let result = queue.append_group(group).await;
            if difference == 0 {
                let _group_lease = result.unwrap();
                assert_eq!(queue.next_batch().await.unwrap().0.len(), 2);
            } else {
                assert!(result.is_err());
                assert!(queue.next_batch().await.is_none());
            }
            queue.shutdown().await;
        }
    }

    #[tokio::test]
    async fn queue_budget_charges_fanout_and_releases_dispatch_and_cancellation() {
        let queue = test_queue_with_expanded_limit(100, None, None, 1000).with_token_limits(Some(
            QueueTokenLimits {
                flex: 10,
                default: 20,
                priority: 30,
            },
        ));
        let (mut a, ar) = block_entry(6, 2);
        let (mut b, br) = block_entry(6, 2);
        a.metadata.service_tier = ServiceTier::Flex;
        b.metadata.service_tier = ServiceTier::Flex;
        let lease = queue.append_group(vec![a, b]).await.unwrap();
        assert_eq!(queue.queued_compute_tokens(), 10);
        let (mut rejected, _rr) = block_entry(1, 1);
        rejected.metadata.service_tier = ServiceTier::Flex;
        assert!(matches!(
            queue.append(rejected).await,
            Err(crate::TextEmbeddingsError::QueueFull { .. })
        ));
        let batch = queue.next_batch().await.unwrap();
        assert_eq!(batch.0.len(), 2);
        // Prepared batches waiting for a replica remain charged.
        assert_eq!(queue.queued_compute_tokens(), 10);
        for metadata in &batch.0 {
            metadata.mark_dispatched();
        }
        assert_eq!(queue.queued_compute_tokens(), 0);
        drop((lease, ar, br, batch));
        let (entry, response) = block_entry(6, 2);
        let lease = queue.append(entry).await.unwrap();
        assert_eq!(queue.queued_compute_tokens(), 8);
        drop((response, lease));
        assert_eq!(queue.queued_compute_tokens(), 0);
        assert!(queue.next_batch().await.is_none());
        let (entry, _response) = block_entry(6, 2);
        let lease = queue.append(entry).await.unwrap();
        let batch = queue.next_batch().await.unwrap();
        assert_eq!(queue.queued_compute_tokens(), 8);
        drop(batch); // Failed dispatch releases tokens even before the caller polls its error.
        assert_eq!(queue.queued_compute_tokens(), 0);
        drop(lease);
        queue.close();
        let (entry, _response) = block_entry(6, 2);
        assert!(queue.append(entry).await.is_err());
        assert_eq!(queue.queued_compute_tokens(), 0);
        queue.shutdown().await;
    }

    #[tokio::test]
    async fn channel_waiters_are_counted_and_cancellation_releases_them() {
        let (sender, _receiver) = mpsc::channel(1);
        let mut queue = test_queue(100);
        queue.shutdown().await;
        queue.closed = Arc::new(AtomicBool::new(false));
        queue.queue_sender = sender;
        queue.queue_sender.send(QueueCommand::Close).await.unwrap();
        let (entry, _response) = test_entry(vec![1; 8], false);
        let q = queue.clone();
        let waiter = tokio::spawn(async move { q.append(entry).await });
        for _ in 0..100 {
            if queue.queued_compute_tokens() == 8 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(queue.queued_compute_tokens(), 8);
        waiter.abort();
        let _ = waiter.await;
        assert_eq!(queue.queued_compute_tokens(), 0);
    }

    #[test]
    fn priority_gets_one_second_of_virtual_age_without_starving_older_work() {
        let mut queue = PendingEntries::new(8, Duration::from_secs(1), Duration::from_secs(1));
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
            entry.metadata.service_tier = if priority {
                ServiceTier::Priority
            } else {
                ServiceTier::Default
            };
            entry.metadata.queue_time = base + Duration::from_millis(millis);
            responses.push(response);
            let now = base + Duration::from_millis(millis);
            queue.push_back(entry, now);
        }
        for id in [1, 2, 4, 5] {
            let PendingEntry { entry, .. } = queue.pop_front().unwrap();
            assert_eq!(entry.encoding.input_ids, [id; 2]);
            if id == 4 {
                assert_eq!(entry.metadata.queue_time, base + Duration::from_secs(2));
            }
        }
        // Later priority arrivals cannot keep passing an older normal input.
        let (mut entry, response) = test_entry(vec![6; 2], true);
        entry.metadata.service_tier = ServiceTier::Priority;
        entry.metadata.queue_time = base + Duration::from_millis(2_501);
        responses.push(response);
        queue.push_back(entry, base + Duration::from_millis(2_501));
        for id in [3, 6] {
            let PendingEntry { entry, .. } = queue.pop_front().unwrap();
            assert_eq!(entry.encoding.input_ids, [id; 2]);
        }
        assert!(queue.pop_front().is_none());
    }

    #[test]
    fn flex_yields_to_newer_work_but_ages_and_preserves_ties() {
        for (bonus, penalty, expected) in [
            (1000, 10000, [3, 2, 1, 4, 5, 6, 7]),
            (0, 0, [1, 2, 3, 4, 5, 6, 7]),
        ] {
            let mut queue = PendingEntries::new(
                8,
                Duration::from_millis(bonus),
                Duration::from_millis(penalty),
            );
            let base = Instant::now();
            for (id, tier, millis) in [
                (1, ServiceTier::Flex, 0),
                (2, ServiceTier::Default, 0),
                (3, ServiceTier::Priority, 0),
                (4, ServiceTier::Default, 10000),
                (5, ServiceTier::Priority, 11000),
                (6, ServiceTier::Default, 11001),
                (7, ServiceTier::Flex, 11001),
            ] {
                let (mut entry, _) = test_entry(vec![id], true);
                entry.metadata.service_tier = tier;
                queue.push_back(entry, base + Duration::from_millis(millis));
            }
            for id in expected {
                let pending = queue.pop_front().unwrap();
                assert_eq!(pending.entry.encoding.input_ids, [id]);
                queue.push_front(pending);
                assert_eq!(queue.pop_front().unwrap().entry.encoding.input_ids, [id]);
            }
            assert!(queue.pop_front().is_none());
        }
    }

    #[test]
    fn configured_bonus_and_zero_fifo_preserve_admission_order() {
        let base = Instant::now();
        for (bonus_ms, expected) in [(0, [1, 2, 3, 4]), (500, [1, 2, 3, 4]), (1000, [2, 1, 4, 3])] {
            let mut queue =
                PendingEntries::new(4, Duration::from_millis(bonus_ms), Duration::from_secs(1));
            for (id, priority, millis) in [
                (1, false, 0),
                (2, true, 750),
                (3, false, 1000),
                (4, true, 1500),
            ] {
                let (mut entry, _) = test_entry(vec![id], true);
                entry.metadata.service_tier = if priority {
                    ServiceTier::Priority
                } else {
                    ServiceTier::Default
                };
                // Producer timestamps can arrive out of order; scheduling uses actor admission.
                entry.metadata.queue_time = base + Duration::from_secs(5 - u64::from(id));
                queue.push_back(entry, base + Duration::from_millis(millis));
            }
            for id in expected {
                let pending = queue.pop_front().unwrap();
                assert_eq!(pending.entry.encoding.input_ids, [id]);
                // A hard batch cut returns the same head without changing its scheduling time.
                queue.push_front(pending);
                assert_eq!(queue.pop_front().unwrap().entry.encoding.input_ids, [id]);
            }
            assert!(queue.pop_front().is_none());
        }
    }

    #[tokio::test]
    async fn tiered_entries_preserve_hard_cuts_and_cancellation() {
        for tier in [ServiceTier::Priority, ServiceTier::Flex] {
            let queue = test_queue(3);
            let mut responses = Vec::new();
            for id in [1, 2, 3] {
                let (mut entry, response) = test_entry(vec![id; 2], true);
                entry.metadata.service_tier = tier;
                let _queue_lease = queue.append(entry).await.unwrap();
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
    }

    #[tokio::test]
    async fn input_admission_survives_queue_and_batch_cancellation() {
        let queue = test_queue(16);
        let budget = Arc::new(tokio::sync::Semaphore::new(1));
        let (canceled, response) = test_entry(vec![1, 2], true);
        let owner = Arc::new(budget.clone().try_acquire_owned().unwrap());
        let _queue_lease = crate::input_memory::with_input_memory(owner, queue.append(canceled))
            .await
            .unwrap();
        drop(response);
        assert_eq!(budget.available_permits(), 0);
        assert!(queue.next_batch().await.is_none());
        assert_eq!(budget.available_permits(), 1);

        let (live, _response) = test_entry(vec![3, 4], true);
        let owner = Arc::new(budget.clone().try_acquire_owned().unwrap());
        let _queue_lease = crate::input_memory::with_input_memory(owner, queue.append(live))
            .await
            .unwrap();
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
    async fn backlog_accounting_preserves_hard_cuts_and_cancellation() {
        let queue = test_queue_with_limits(15_000, None, Some(5_000));
        let mut responses = Vec::new();
        for value in 0..4 {
            let (entry, response) = test_entry(vec![value; 5_000], true);
            responses.push(response);
            let _queue_lease = queue.append(entry).await.unwrap();
        }
        // The original 20k backlog disables the soft target for the entire batch.
        let (metadata, batch) = queue.next_batch().await.unwrap();
        assert_eq!(metadata.len(), 3);
        assert_eq!(batch.input_ids.len(), 15_000);
        assert_eq!(queue.next_batch().await.unwrap().0.len(), 1);
        assert!(queue.next_batch().await.is_none());

        let (canceled, response) = test_entry(vec![9; 15_000], true);
        drop(response);
        let _queue_lease = queue.append(canceled).await.unwrap();
        let (live, response) = test_entry(vec![10; 5_000], true);
        responses.push(response);
        let _queue_lease = queue.append(live).await.unwrap();
        let (live, response) = test_entry(vec![10; 5_000], true);
        responses.push(response);
        let _queue_lease = queue.append(live).await.unwrap();
        // The canceled input conservatively disables only this soft cut.
        assert_eq!(queue.next_batch().await.unwrap().0.len(), 2);
        assert!(queue.next_batch().await.is_none());
        // Neither a hard-cut front entry nor a canceled entry leaves a stale charge.
        for value in 11..14 {
            let (entry, response) = test_entry(vec![value; 5_000], true);
            responses.push(response);
            let _queue_lease = queue.append(entry).await.unwrap();
        }
        for value in 11..14 {
            let (metadata, batch) = queue.next_batch().await.unwrap();
            assert_eq!(metadata.len(), 1);
            assert_eq!(batch.input_ids, vec![value; 5_000]);
        }
        assert!(queue.next_batch().await.is_none());
        queue.close();
    }

    #[tokio::test]
    async fn request_limit_keeps_remaining_tokens_in_the_snapshot() {
        let queue = test_queue_with_limits(32_000, Some(2), Some(5_000));
        let mut responses = Vec::new();
        for value in 0..6 {
            let (entry, response) = test_entry(vec![value; 5_000], true);
            responses.push(response);
            let _queue_lease = queue.append(entry).await.unwrap();
        }
        assert_eq!(queue.next_batch().await.unwrap().0.len(), 2);
        // Twenty thousand tokens remain, so the next selection still bypasses
        // the soft target while respecting the two-request hard limit.
        assert_eq!(queue.next_batch().await.unwrap().0.len(), 2);
        assert_eq!(queue.next_batch().await.unwrap().0.len(), 1);
        assert_eq!(queue.next_batch().await.unwrap().0.len(), 1);
        assert!(queue.next_batch().await.is_none());
        queue.close();
    }

    #[tokio::test]
    async fn lookahead_preserves_interleaved_clients_and_cancellation() {
        for cancel_grouped_tail in [false, true] {
            let queue = test_queue_with_limits(32, None, Some(3));
            let client = Arc::new(AtomicUsize::new(0));
            let mut responses = Vec::new();
            for value in 1..5 {
                let (mut entry, response) = test_entry(vec![value; 3], true);
                entry.metadata.client_batch = (value == 1 || value == 3).then(|| client.clone());
                if cancel_grouped_tail && value == 3 {
                    drop(response);
                } else {
                    responses.push(response);
                }
                let _queue_lease = queue.append(entry).await.unwrap();
            }
            let (metadata, batch) = queue.next_batch().await.unwrap();
            assert_eq!(metadata.len(), if cancel_grouped_tail { 2 } else { 3 });
            assert_eq!(&batch.input_ids[..6], &[1, 1, 1, 2, 2, 2]);
            // Removing the tail also removes its lookahead charge; the final
            // unrelated input starts a new selection in both cases.
            let (metadata, batch) = queue.next_batch().await.unwrap();
            assert_eq!(metadata.len(), 1);
            assert_eq!(batch.input_ids, [4, 4, 4]);
            assert!(queue.next_batch().await.is_none());
            queue.close();
        }
    }

    #[tokio::test]
    async fn lookahead_matches_fifo_reference_across_client_patterns() {
        for scenario in 0..256usize {
            let clients: Vec<_> = (0..4)
                .map(|i| Arc::new(AtomicUsize::new(usize::from((scenario + i) % 7 == 0))))
                .collect();
            let max_tokens = 32 + scenario % 64;
            let max_requests = (scenario % 3 == 0).then_some(1 + scenario % 7);
            let target = (scenario % 4 != 0).then_some(8 + scenario % 24);
            let queue = test_queue_with_limits(max_tokens, max_requests, target);
            let mut reference = VecDeque::new();
            let mut responses = Vec::new();
            let mut seed = scenario as u64 + 1;
            for id in 0..32 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let tokens = 1 + ((seed >> 32) % 16) as usize;
                let group = ((seed >> 40) % 5) as usize;
                let (mut entry, response) = test_entry(vec![id; tokens], true);
                entry.metadata.client_batch = clients.get(group).cloned();
                let _queue_lease = queue.append(entry).await.unwrap();
                responses.push(response);
                reference.push_back((id, tokens, group));
            }
            while !reference.is_empty() {
                let mut expected: Vec<(u32, usize, usize)> = Vec::new();
                let mut tokens = 0;
                while let Some(&(id, count, group)) = reference.front() {
                    let unfinished = expected.iter().any(|&(_, _, selected_group)| {
                        clients.get(selected_group).is_some_and(|client| {
                            client.load(Ordering::SeqCst) != 0
                                || selected_group == group
                                || reference
                                    .iter()
                                    .any(|&(_, _, pending_group)| pending_group == selected_group)
                        })
                    });
                    if should_cut(tokens, count, target) && !unfinished {
                        break;
                    }
                    if tokens + count > max_tokens {
                        break;
                    }
                    reference.pop_front();
                    expected.push((id, count, group));
                    tokens += count;
                    if Some(expected.len()) == max_requests {
                        break;
                    }
                }
                let (metadata, batch) = queue.next_batch().await.unwrap();
                assert_eq!(metadata.len(), expected.len(), "scenario {scenario}");
                assert_eq!(batch.input_ids.len(), tokens, "scenario {scenario}");
                assert_eq!(
                    batch.cumulative_seq_lengths[..metadata.len()]
                        .iter()
                        .map(|&offset| batch.input_ids[offset as usize])
                        .collect::<Vec<_>>(),
                    expected.iter().map(|&(id, _, _)| id).collect::<Vec<_>>(),
                    "scenario {scenario}"
                );
            }
            assert!(queue.next_batch().await.is_none());
            queue.close();
        }
    }

    #[tokio::test]
    async fn staged_buffers_preserve_single_inputs_and_pending_batches() {
        let queue = test_queue(32);
        assert!(queue.next_batch().await.is_none());
        let (canceled, response) = test_entry(vec![0; 2], true);
        drop(response);
        let _queue_lease = queue.append(canceled).await.unwrap();
        assert!(queue.next_batch().await.is_none());

        let (single, _response) = test_entry(vec![10, 11], true);
        let pointers = (
            single.encoding.input_ids.as_ptr(),
            single.encoding.token_type_ids.as_ptr(),
            single.encoding.position_ids.as_ptr(),
        );
        let _queue_lease = queue.append(single).await.unwrap();
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
            let _queue_lease = queue.append(first).await.unwrap();
            let _queue_lease = queue.append(second).await.unwrap();
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
                service_tier: ServiceTier::Default,
                input_memory: None,
                queue_reservation: None,
                client_batch: None,
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
        assert_eq!(target_for_backlog(100_000, Some(5_000), 8), Some(5_000));
        assert_eq!(target_for_backlog(160_000, Some(5_000), 8), None);
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
                service_tier: ServiceTier::Default,
                input_memory: None,
                queue_reservation: None,
                client_batch: None,
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
