// Owning Qwen3-VL engine for the normal BackendThread.
//
// One model load, one CUDA stream, exactly two request-specific pinned HostIo
// slots, per-request workspaces and same-stream completion tickets. The complete
// RAII owner is constructed before `Inference` is returned, and the deferred
// closure captures that one owner. Dropping the Inference without invoking the
// closure runs the owner Drop, which fences or quarantines every resource.

use crate::checkpoint::start;
use crate::qwen3_vl_batch::PackedVlRequest;
use crate::qwen3_vl_service::{prepare_batch, Qwen3VlModel};
use crate::staging::{HostIo, HostIoLease, SlotPool};
use crate::work_retirement::CompletionTicket;
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::{BackendError, Batch, Embeddings, Inference, Pool};
use std::sync::{Arc, Mutex};

fn infer(error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide Qwen3-VL engine: {error}"))
}

fn product(left: usize, right: usize) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| infer("dimension overflow"))
}

struct WorkspaceCache {
    free: Vec<crate::qwen3_vl_device::Qwen3VlWorkspaces>,
    retired: Vec<(CompletionTicket, crate::qwen3_vl_device::Qwen3VlWorkspaces)>,
    retained_bytes: usize,
    budget: usize,
}

impl WorkspaceCache {
    fn new(budget: usize) -> Self {
        Self {
            free: Vec::new(),
            retired: Vec::new(),
            retained_bytes: 0,
            budget,
        }
    }

    fn take(&mut self) -> Result<crate::qwen3_vl_device::Qwen3VlWorkspaces, BackendError> {
        self.drain();
        if let Some(workspaces) = self.free.pop() {
            self.retained_bytes = self.retained_bytes.saturating_sub(workspaces.bytes());
            return Ok(workspaces);
        }
        if self.retained_bytes >= self.budget {
            return Err(BackendError::Overloaded(
                "Qwen3-VL workspace byte budget exhausted".into(),
            ));
        }
        Ok(Qwen3VlModel::workspace())
    }

    fn retire(
        &mut self,
        ticket: CompletionTicket,
        workspaces: crate::qwen3_vl_device::Qwen3VlWorkspaces,
    ) {
        let bytes = workspaces.bytes();
        if bytes > self.budget {
            // The workspace was already fenced; do not pin an oversized retired
            // allocation. Drop it instead of retaining it.
            return;
        }
        self.retained_bytes = self.retained_bytes.saturating_add(bytes);
        self.retired.push((ticket, workspaces));
        self.trim();
    }

    fn drain(&mut self) {
        let mut index = 0;
        while index < self.retired.len() {
            if self.retired[index].0.completed().is_some() {
                let (_, workspaces) = self.retired.swap_remove(index);
                self.free.push(workspaces);
            } else {
                index += 1;
            }
        }
    }

    fn trim(&mut self) {
        while self.retained_bytes > self.budget {
            let Some(workspaces) = self.free.pop() else {
                break;
            };
            self.retained_bytes = self.retained_bytes.saturating_sub(workspaces.bytes());
        }
    }
}

trait FencedLease {
    fn wait(&mut self) -> Result<(), BackendError>;
    fn quarantine(self);
}

impl FencedLease for HostIoLease {
    fn wait(&mut self) -> Result<(), BackendError> {
        HostIoLease::wait(self)
    }

    fn quarantine(self) {
        HostIoLease::quarantine(self)
    }
}

trait RetiredResource {
    fn retire(&mut self);
}

struct VlResources {
    model: Arc<Mutex<Qwen3VlModel>>,
    request: PackedVlRequest,
    workspaces: Option<crate::qwen3_vl_device::Qwen3VlWorkspaces>,
    ticket: Option<CompletionTicket>,
    cache: Arc<Mutex<WorkspaceCache>>,
    batch: Batch,
    pool: Pool,
    raw_tokens: bool,
    pending: Option<crate::qwen3_vl_device::Qwen3VlPending>,
}

impl RetiredResource for VlResources {
    fn retire(&mut self) {
        if let (Some(ticket), Some(workspaces)) = (self.ticket.take(), self.workspaces.take()) {
            if let Ok(mut cache) = self.cache.lock() {
                cache.retire(ticket, workspaces);
            } else {
                // The cache is poisoned; retain the device workspaces rather
                // than freeing memory whose fence we cannot record.
                std::mem::forget(workspaces);
            }
        }
    }
}

impl VlResources {
    fn pack(&self, raw: &[f32]) -> Result<Embeddings, BackendError> {
        self.model
            .lock()
            .map_err(|_| infer("model lock is poisoned"))?
            .pack_raw(&self.batch, self.pool.clone(), self.raw_tokens, raw)
    }
}

struct FencedOwner<L: FencedLease, R: RetiredResource> {
    lease: Option<L>,
    resources: Option<R>,
    completed: bool,
}

impl<L: FencedLease, R: RetiredResource> FencedOwner<L, R> {
    fn new(lease: L, resources: R) -> Self {
        Self {
            lease: Some(lease),
            resources: Some(resources),
            completed: false,
        }
    }

    fn wait(&mut self) -> Result<(), BackendError> {
        self.lease
            .as_mut()
            .ok_or_else(|| infer("missing VL host lease"))?
            .wait()
    }

    fn finish(&mut self) {
        self.completed = true;
        if let Some(mut resources) = self.resources.take() {
            resources.retire();
        }
        if let Some(lease) = self.lease.take() {
            drop(lease);
        }
    }

    fn fail_closed(&mut self) {
        self.completed = true;
        if let Some(lease) = self.lease.take() {
            lease.quarantine();
        }
        if let Some(resources) = self.resources.take() {
            std::mem::forget(resources);
        }
    }
}

impl<L: FencedLease, R: RetiredResource> Drop for FencedOwner<L, R> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if self.wait().is_ok() {
            self.finish();
        } else {
            // A failed/uncertain fence must retain model, request, workspaces and
            // the pinned host slot together, and fail the engine closed.
            self.fail_closed();
        }
    }
}

impl FencedOwner<HostIoLease, VlResources> {
    fn output(&self, elements: usize) -> Result<&[f32], BackendError> {
        self.lease
            .as_ref()
            .ok_or_else(|| infer("missing VL host lease"))?
            .output(elements)
    }

    fn launch(&mut self) -> Result<(), BackendError> {
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| infer("missing VL resources"))?;
        let lease = self
            .lease
            .as_mut()
            .ok_or_else(|| infer("missing VL host lease"))?;
        let mut model = resources
            .model
            .lock()
            .map_err(|_| infer("model lock is poisoned"))?;
        let workspaces = resources
            .workspaces
            .as_mut()
            .ok_or_else(|| infer("missing VL workspaces"))?;
        let pending = model.launch_packed(&resources.request, workspaces, lease)?;
        resources.pending = Some(pending);
        Ok(())
    }

    fn record(&mut self) -> Result<(), BackendError> {
        self.lease
            .as_mut()
            .ok_or_else(|| infer("missing VL host lease"))?
            .record()
    }

    fn complete(&mut self) -> Result<Embeddings, BackendError> {
        self.wait()?;
        let elements = self
            .resources
            .as_ref()
            .and_then(|resources| resources.pending)
            .ok_or_else(|| infer("missing VL pending shape"))?
            .elements;
        let raw = self.output(elements)?.to_vec();
        let result = self
            .resources
            .as_ref()
            .ok_or_else(|| infer("missing VL resources"))?
            .pack(&raw);
        self.finish();
        result
    }
}

pub(crate) struct Qwen3VlEngine {
    context: Arc<Context>,
    stream: Arc<Stream>,
    model: Arc<Mutex<Qwen3VlModel>>,
    io: Arc<SlotPool<HostIo>>,
    workspaces: Arc<Mutex<WorkspaceCache>>,
    pool: Pool,
}

impl Qwen3VlEngine {
    pub(crate) fn new(
        model: Qwen3VlModel,
        context: Arc<Context>,
        pool: Pool,
    ) -> Result<Self, BackendError> {
        let stream = model.stream();
        let io = HostIo::pool(&context)?;
        Ok(Self {
            context,
            stream,
            model: Arc::new(Mutex::new(model)),
            io,
            workspaces: Arc::new(Mutex::new(WorkspaceCache::new(crate::cache::BYTE_BUDGET))),
            pool,
        })
    }

    pub(crate) fn output_width(&self) -> Result<usize, BackendError> {
        self.model
            .lock()
            .map(|model| model.output_width())
            .map_err(|_| infer("model lock is poisoned"))
    }

    pub(crate) fn config(&self) -> Result<crate::qwen3_vl_config::Qwen3VlConfig, BackendError> {
        self.model
            .lock()
            .map(|model| model.config().clone())
            .map_err(|_| infer("model lock is poisoned"))
    }

    pub(crate) fn health(&self) -> Result<(), BackendError> {
        if self.io.poisoned() {
            return Err(BackendError::Unhealthy);
        }
        self.stream.synchronize().map_err(start)
    }

    pub(crate) fn submit(&self, batch: Batch) -> Result<Inference<Embeddings>, BackendError> {
        self.submit_with_pool(batch, self.pool.clone())
    }

    pub(crate) fn submit_with_pool(
        &self,
        batch: Batch,
        pool: Pool,
    ) -> Result<Inference<Embeddings>, BackendError> {
        if batch.input_ids.is_empty() {
            return Err(infer("empty Qwen3-VL input"));
        }
        let (multi_vector, hidden, config) = {
            let model = self
                .model
                .lock()
                .map_err(|_| infer("model lock is poisoned"))?;
            (
                model.supports_multi_vector(),
                model.hidden(),
                model.config().clone(),
            )
        };
        if multi_vector && !batch.pooled_indices.is_empty() {
            return Err(infer(
                "Qwen3-VL multi-vector wrappers return token rows and reject pooling",
            ));
        }
        // Validate every media descriptor before the empty-output shortcut,
        // before any lease or ticket. Invalid media with zero selectors is
        // still rejected and remains recoverable.
        let request: PackedVlRequest = prepare_batch(&config, &batch, self.pool.clone(), true)?;
        if batch.pooled_indices.is_empty() && batch.raw_indices.is_empty() {
            return Ok(Inference::Ready(Embeddings::default()));
        }
        let tokens = batch.input_ids.len();
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| infer("missing cumulative boundaries"))?;
        if sequences == 0 {
            return Err(infer("empty sequence count"));
        }
        let output_elements = product(tokens, hidden)?;
        let key = (tokens, batch.cumulative_seq_lengths.len(), tokens);
        let slot = self.io.acquire()?;
        let mut lease = HostIoLease::new_detached(slot, self.io.clone(), self.stream.clone());
        lease.prepare(key, output_elements, false)?;
        // One context bind per submission; no bind inside the forward or per op.
        self.context.bind_to_thread().map_err(start)?;
        let ticket = lease.begin_tracked()?;
        let workspaces = self
            .workspaces
            .lock()
            .map_err(|_| infer("workspace cache is poisoned"))?
            .take()?;
        let raw_tokens = multi_vector || !batch.raw_indices.is_empty();
        let resources = VlResources {
            model: Arc::clone(&self.model),
            request,
            workspaces: Some(workspaces),
            ticket: Some(ticket),
            cache: Arc::clone(&self.workspaces),
            batch,
            pool,
            raw_tokens,
            pending: None,
        };
        // Construct the complete owner BEFORE returning Inference. The closure
        // captures exactly this owner, so dropping Inference runs its Drop.
        let mut owner = FencedOwner::new(lease, resources);
        owner.launch()?;
        owner.record()?;
        Ok(Inference::Deferred(Box::new(move || owner.complete())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work_retirement::CompletionClock;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn workspace_cache_holds_ab_until_the_ticket_is_fenced_then_reuses() {
        let clock = Arc::new(CompletionClock::new());
        let ticket_a = clock.ticket(7).unwrap();
        let ticket_b = clock.ticket(7).unwrap();
        let mut cache = WorkspaceCache::new(crate::cache::BYTE_BUDGET);
        let a = cache.take().unwrap();
        let b = cache.take().unwrap();
        cache.retire(ticket_a.clone(), a);
        cache.retire(ticket_b.clone(), b);
        let _c = cache.take().unwrap();
        assert_eq!(cache.retired.len(), 2);
        assert_eq!(cache.free.len(), 0);
        unsafe { ticket_a.mark_completed_after_fence() };
        cache.drain();
        assert_eq!(cache.retired.len(), 1);
        assert_eq!(cache.free.len(), 1);
        unsafe { ticket_b.mark_completed_after_fence() };
        cache.drain();
        assert_eq!(cache.retired.len(), 0);
        assert_eq!(cache.free.len(), 2);
    }

    #[derive(Default)]
    struct Log(StdMutex<Vec<&'static str>>);

    impl Log {
        fn push(&self, value: &'static str) {
            self.0.lock().unwrap().push(value);
        }
        fn values(&self) -> Vec<&'static str> {
            self.0.lock().unwrap().clone()
        }
    }

    struct FakeLease {
        fence: bool,
        log: Arc<Log>,
    }

    impl FencedLease for FakeLease {
        fn wait(&mut self) -> Result<(), BackendError> {
            self.log.push("wait");
            if self.fence {
                Ok(())
            } else {
                Err(BackendError::Unhealthy)
            }
        }

        fn quarantine(self) {
            self.log.push("quarantine");
        }
    }

    struct FakeResources {
        log: Arc<Log>,
    }

    impl RetiredResource for FakeResources {
        fn retire(&mut self) {
            self.log.push("retire");
        }
    }

    impl Drop for FakeResources {
        fn drop(&mut self) {
            self.log.push("resource_drop");
        }
    }

    #[test]
    fn dropping_inference_without_completion_fences_before_resource_drop() {
        let log = Arc::new(Log::default());
        {
            let owner = FencedOwner::new(
                FakeLease {
                    fence: true,
                    log: Arc::clone(&log),
                },
                FakeResources {
                    log: Arc::clone(&log),
                },
            );
            drop(owner);
        }
        assert_eq!(log.values(), vec!["wait", "retire", "resource_drop"]);
    }

    #[test]
    fn failed_fence_quarantines_lease_and_retains_resources() {
        let log = Arc::new(Log::default());
        {
            let owner = FencedOwner::new(
                FakeLease {
                    fence: false,
                    log: Arc::clone(&log),
                },
                FakeResources {
                    log: Arc::clone(&log),
                },
            );
            drop(owner);
        }
        let values = log.values();
        assert_eq!(values, vec!["wait", "quarantine"]);
        assert!(!values.contains(&"resource_drop"));
    }

    #[test]
    fn partial_enqueue_then_failed_fence_retains_all_owners() {
        let log = Arc::new(Log::default());
        {
            // Simulate the state after launch enqueued work and record failed:
            // the owner is dropped with an unproven fence.
            let owner = FencedOwner::new(
                FakeLease {
                    fence: false,
                    log: Arc::clone(&log),
                },
                FakeResources {
                    log: Arc::clone(&log),
                },
            );
            drop(owner);
        }
        assert_eq!(log.values(), vec!["wait", "quarantine"]);
    }

    #[test]
    fn workspace_byte_budget_fails_before_allocating_another_owner() {
        let mut cache = WorkspaceCache::new(1);
        cache.retained_bytes = 1;
        assert!(matches!(cache.take(), Err(BackendError::Overloaded(_))));
    }
}
