// Owning Qwen3-VL engine for the normal BackendThread.
//
// One model load, one CUDA stream, exactly two request-specific pinned HostIo
// slots, per-request workspaces and same-stream completion tickets. Submission
// returns `Inference::Deferred`; the forward is not a synchronous Ready value.

use crate::checkpoint::start;
use crate::qwen3_vl_batch::{pack_batch, PackedVlRequest};
use crate::qwen3_vl_service::Qwen3VlModel;
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

#[derive(Default)]
struct WorkspaceCache {
    free: Vec<crate::qwen3_vl_device::Qwen3VlWorkspaces>,
    retired: Vec<(CompletionTicket, crate::qwen3_vl_device::Qwen3VlWorkspaces)>,
}

impl WorkspaceCache {
    fn take(&mut self) -> crate::qwen3_vl_device::Qwen3VlWorkspaces {
        self.drain();
        self.free.pop().unwrap_or_else(Qwen3VlModel::workspace)
    }

    fn retire(
        &mut self,
        ticket: CompletionTicket,
        workspaces: crate::qwen3_vl_device::Qwen3VlWorkspaces,
    ) {
        self.retired.push((ticket, workspaces));
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
}

/// Deferred owner for one in-flight request. Drop always waits for the
/// same-stream fence before the workspaces are freed or retired, including the
/// case where the caller drops the `Inference` without completion.
struct DeferredVl {
    lease: Option<HostIoLease>,
    workspaces: Option<crate::qwen3_vl_device::Qwen3VlWorkspaces>,
    ticket: Option<CompletionTicket>,
    cache: Arc<Mutex<WorkspaceCache>>,
}

impl DeferredVl {
    fn wait(&mut self) -> Result<(), BackendError> {
        self.lease
            .as_mut()
            .ok_or_else(|| infer("missing VL host lease"))?
            .wait()
    }

    fn output(&self, elements: usize) -> Result<&[f32], BackendError> {
        self.lease
            .as_ref()
            .ok_or_else(|| infer("missing VL host lease"))?
            .output(elements)
    }

    fn finish(&mut self) {
        if let Some(lease) = self.lease.take() {
            drop(lease);
        }
        if let (Some(ticket), Some(workspaces)) = (self.ticket.take(), self.workspaces.take()) {
            if let Ok(mut cache) = self.cache.lock() {
                cache.retire(ticket, workspaces);
            } else {
                std::mem::forget(workspaces);
            }
        }
    }
}

impl Drop for DeferredVl {
    fn drop(&mut self) {
        if self.lease.is_some() {
            let fenced = self.wait().is_ok();
            if !fenced {
                if let Some(workspaces) = self.workspaces.take() {
                    std::mem::forget(workspaces);
                }
            }
        }
        self.finish();
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
            workspaces: Arc::new(Mutex::new(WorkspaceCache::default())),
            pool,
        })
    }

    pub(crate) fn config(&self) -> Result<crate::qwen3_vl_config::Qwen3VlConfig, BackendError> {
        self.model
            .lock()
            .map(|model| model.config().clone())
            .map_err(|_| infer("model lock is poisoned"))
    }

    pub(crate) fn health(&self) -> Result<(), BackendError> {
        self.stream.synchronize().map_err(start)
    }

    pub(crate) fn submit(&self, batch: Batch) -> Result<Inference<Embeddings>, BackendError> {
        if batch.input_ids.is_empty() {
            return Err(infer("empty Qwen3-VL input"));
        }
        let (multi_vector, hidden) = {
            let model = self
                .model
                .lock()
                .map_err(|_| infer("model lock is poisoned"))?;
            (model.supports_multi_vector(), model.hidden())
        };
        if multi_vector && !batch.pooled_indices.is_empty() {
            return Err(infer(
                "Qwen3-VL multi-vector wrappers return token rows and reject pooling",
            ));
        }
        if batch.pooled_indices.is_empty() && batch.raw_indices.is_empty() {
            return Ok(Inference::Ready(Embeddings::default()));
        }
        // Validate every media descriptor before touching a lease or a ticket.
        // A malformed image is a recoverable inference error and leaves the
        // engine/workspaces healthy.
        let request: PackedVlRequest = pack_batch(&batch, self.pool.clone(), true)?;
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
        // Ticket is issued before the first GPU upload for this request.
        let ticket = lease.begin_tracked()?;
        let mut workspaces = self
            .workspaces
            .lock()
            .map_err(|_| infer("workspace cache is poisoned"))?
            .take();
        let pending = {
            let mut model = self
                .model
                .lock()
                .map_err(|_| infer("model lock is poisoned"))?;
            match model.launch_packed(&request, &mut workspaces, &mut lease) {
                Ok(pending) => pending,
                Err(error) => {
                    // A launch failure may follow partial enqueues. Drain or
                    // retain uncertain memory, and drop the lease before the
                    // workspaces it may still reference.
                    let _ = lease.wait();
                    drop(lease);
                    drop(workspaces);
                    return Err(error);
                }
            }
        };
        if let Err(error) = lease.record() {
            let _ = lease.wait();
            drop(lease);
            std::mem::forget(workspaces);
            return Err(error);
        }
        let raw_tokens = multi_vector || !batch.raw_indices.is_empty();
        let model = Arc::clone(&self.model);
        let cache = Arc::clone(&self.workspaces);
        let pool = self.pool.clone();
        Ok(Inference::Deferred(Box::new(move || {
            // Retain the packed request (image pixels, ids, positions) until
            // the same-stream fence; async H2D copies still reference it.
            let _request = request;
            let mut deferred = DeferredVl {
                lease: Some(lease),
                workspaces: Some(workspaces),
                ticket: Some(ticket),
                cache,
            };
            deferred.wait()?;
            let result = (|| -> Result<Embeddings, BackendError> {
                let raw = deferred.output(pending.elements)?.to_vec();
                model
                    .lock()
                    .map_err(|_| infer("model lock is poisoned"))?
                    .pack_raw(&batch, pool, raw_tokens, &raw)
            })();
            deferred.finish();
            result
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work_retirement::CompletionClock;

    #[test]
    fn workspace_cache_holds_ab_until_the_ticket_is_fenced_then_reuses() {
        let clock = Arc::new(CompletionClock::new());
        let ticket_a = clock.ticket(7).unwrap();
        let ticket_b = clock.ticket(7).unwrap();
        let mut cache = WorkspaceCache::default();
        let a = cache.take();
        let b = cache.take();
        cache.retire(ticket_a.clone(), a);
        cache.retire(ticket_b.clone(), b);
        // Both are still in flight: a third request cannot reuse either.
        let _c = cache.take();
        assert_eq!(cache.retired.len(), 2);
        // C is held by the third request, not by the cache.
        assert_eq!(cache.free.len(), 0);
        // One same-stream fence completes A only; A is reusable, B is held.
        unsafe { ticket_a.mark_completed_after_fence() };
        cache.drain();
        assert_eq!(cache.retired.len(), 1);
        assert_eq!(cache.free.len(), 1);
        unsafe { ticket_b.mark_completed_after_fence() };
        cache.drain();
        assert_eq!(cache.retired.len(), 0);
        assert_eq!(cache.free.len(), 2);
    }
}
