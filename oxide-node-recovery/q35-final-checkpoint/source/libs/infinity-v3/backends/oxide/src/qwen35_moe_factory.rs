//! Normal service-factory owner for the actual Qwen3.5-MoE text backbone.
//!
//! This is a thin owning adapter over the tested `ExperimentalMoEEngine`. It
//! does not create a CUDA context or upload weights per request; one engine is
//! constructed at cold startup and owns the stream, weights, two pooled host-I/O
//! leases and request Work arena. Deferred inference acquires one lease and
//! reuses one Work buffer when the request geometry matches.

use crate::output;
use crate::qwen35_moe_engine::{
    ExperimentalMoEEngine, ExperimentalMoELease, ExperimentalMoEOptions, ExperimentalMoEWork,
};
use infinity_v3_backend_core::{
    result_memory, Backend, BackendError, Batch, Embeddings, Inference, ModelType, Pool,
    Predictions, TokenPredictions,
};
use serde_json::Value;
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub fn is_qwen35_moe_checkpoint(path: &Path) -> Result<bool, BackendError> {
    let value: Value = serde_json::from_slice(
        &std::fs::read(path.join("config.json"))
            .map_err(|error| BackendError::Start(format!("read config.json: {error}")))?,
    )
    .map_err(|error| BackendError::Start(format!("parse config.json: {error}")))?;
    let root = value.get("model_type").and_then(Value::as_str);
    let text = value
        .get("text_config")
        .and_then(Value::as_object)
        .and_then(|text| text.get("model_type"))
        .and_then(Value::as_str);
    Ok(matches!(root, Some("qwen3_5_moe" | "qwen3_5_moe_text"))
        || matches!(text, Some("qwen3_5_moe_text")))
}

pub struct Qwen35MoeBackend {
    engine: Arc<ExperimentalMoEEngine>,
    pool: Pool,
    work: Arc<Mutex<WorkSlots<ExperimentalMoEWork>>>,
}

const WORK_SLOTS: usize = 2;

trait WorkBytes {
    fn work_bytes(&self) -> usize;
}

impl WorkBytes for ExperimentalMoEWork {
    fn work_bytes(&self) -> usize {
        self.bytes()
    }
}

enum WorkSlot<W> {
    Empty,
    Building {
        key: (usize, usize, usize),
        bytes: usize,
    },
    Ready {
        key: (usize, usize, usize),
        work: W,
    },
    CheckedOut {
        key: (usize, usize, usize),
        bytes: usize,
    },
    Retained {
        key: (usize, usize, usize),
        bytes: usize,
    },
}

struct WorkSlots<W> {
    slots: Vec<WorkSlot<W>>,
    accounted_bytes: usize,
    byte_budget: usize,
}

impl<W> Default for WorkSlots<W> {
    fn default() -> Self {
        Self {
            slots: (0..WORK_SLOTS).map(|_| WorkSlot::Empty).collect(),
            accounted_bytes: 0,
            byte_budget: crate::cache::BYTE_BUDGET,
        }
    }
}

impl<W: WorkBytes> WorkSlots<W> {
    #[cfg(test)]
    fn with_budget(byte_budget: usize) -> Self {
        Self {
            slots: (0..WORK_SLOTS).map(|_| WorkSlot::Empty).collect(),
            accounted_bytes: 0,
            byte_budget,
        }
    }

    #[cfg(test)]
    fn resident_bytes(&self) -> usize {
        self.accounted_bytes
    }

    fn empty_index(&self) -> Option<usize> {
        self.slots
            .iter()
            .position(|slot| matches!(slot, WorkSlot::Empty))
    }

    fn projected_fits(&self, projected: usize) -> bool {
        if projected > self.byte_budget {
            self.accounted_bytes == 0
        } else {
            self.accounted_bytes
                .checked_add(projected)
                .is_some_and(|total| total <= self.byte_budget)
        }
    }

    fn checkout_ready(&mut self, key: (usize, usize, usize)) -> Option<(usize, W)> {
        for index in 0..self.slots.len() {
            if matches!(self.slots[index], WorkSlot::Ready { key: slot_key, .. } if slot_key == key)
            {
                let slot = std::mem::replace(&mut self.slots[index], WorkSlot::Empty);
                if let WorkSlot::Ready { work, .. } = slot {
                    let bytes = work.work_bytes();
                    self.slots[index] = WorkSlot::CheckedOut { key, bytes };
                    return Some((index, work));
                }
            }
        }
        None
    }

    #[cfg(test)]
    fn install_ready(
        &mut self,
        index: usize,
        key: (usize, usize, usize),
        work: W,
    ) -> Result<(), W> {
        if index >= self.slots.len() || !matches!(self.slots[index], WorkSlot::Empty) {
            return Err(work);
        }
        let bytes = work.work_bytes();
        let Some(total) = self.accounted_bytes.checked_add(bytes) else {
            return Err(work);
        };
        self.accounted_bytes = total;
        self.slots[index] = WorkSlot::Ready { key, work };
        Ok(())
    }

    fn begin_build(
        &mut self,
        index: usize,
        key: (usize, usize, usize),
        projected: usize,
    ) -> Result<(), BackendError> {
        if index >= self.slots.len() || !matches!(self.slots[index], WorkSlot::Empty) {
            return Err(BackendError::Unhealthy);
        }
        let total = self.accounted_bytes.checked_add(projected).ok_or_else(|| {
            BackendError::Overloaded("Qwen3.5-MoE Work byte budget overflow".into())
        })?;
        self.accounted_bytes = total;
        self.slots[index] = WorkSlot::Building {
            key,
            bytes: projected,
        };
        Ok(())
    }

    fn finish_build(
        &mut self,
        index: usize,
        key: (usize, usize, usize),
        projected: usize,
        actual: usize,
    ) -> bool {
        let Some(other) = self.accounted_bytes.checked_sub(projected) else {
            return false;
        };
        let Some(total) = other.checked_add(actual) else {
            return false;
        };
        let fits = if actual > self.byte_budget {
            other == 0
        } else {
            total <= self.byte_budget
        };
        if !fits
            || !matches!(
                self.slots.get(index),
                Some(WorkSlot::Building { key: slot_key, bytes }) if *slot_key == key && *bytes == projected
            )
        {
            return false;
        }
        self.accounted_bytes = total;
        self.slots[index] = WorkSlot::CheckedOut { key, bytes: actual };
        true
    }

    fn abort_build(&mut self, index: usize, key: (usize, usize, usize), projected: usize) {
        if matches!(
            self.slots.get(index),
            Some(WorkSlot::Building { key: slot_key, bytes }) if *slot_key == key && *bytes == projected
        ) {
            if let Some(total) = self.accounted_bytes.checked_sub(projected) {
                self.accounted_bytes = total;
                self.slots[index] = WorkSlot::Empty;
            }
        }
    }

    fn return_ready(&mut self, index: usize, key: (usize, usize, usize), work: W) -> Result<(), W> {
        let valid = matches!(
            self.slots.get(index),
            Some(WorkSlot::CheckedOut { key: slot_key, .. }) if *slot_key == key
        );
        if !valid {
            return Err(work);
        }
        self.slots[index] = WorkSlot::Ready { key, work };
        Ok(())
    }

    fn release_checked_out(&mut self, index: usize, key: (usize, usize, usize)) {
        if let Some(WorkSlot::CheckedOut {
            key: slot_key,
            bytes,
        }) = self.slots.get(index)
        {
            if *slot_key == key {
                let bytes = *bytes;
                if let Some(total) = self.accounted_bytes.checked_sub(bytes) {
                    self.accounted_bytes = total;
                    self.slots[index] = WorkSlot::Empty;
                }
            }
        }
    }

    fn retain_checked_out(&mut self, index: usize, key: (usize, usize, usize)) {
        if let Some(WorkSlot::CheckedOut {
            key: slot_key,
            bytes,
        }) = self.slots.get(index)
        {
            if *slot_key == key {
                let bytes = *bytes;
                self.slots[index] = WorkSlot::Retained { key, bytes };
            }
        }
    }

    fn evict_idle(
        &mut self,
        mut release: impl FnMut(&mut W) -> Result<bool, BackendError>,
    ) -> Result<Option<usize>, BackendError> {
        for index in 0..self.slots.len() {
            let released = match self.slots.get_mut(index) {
                Some(WorkSlot::Ready { work, .. }) => release(work)?,
                _ => false,
            };
            if released {
                let slot = std::mem::replace(&mut self.slots[index], WorkSlot::Empty);
                if let WorkSlot::Ready { work, .. } = slot {
                    if let Some(total) = self.accounted_bytes.checked_sub(work.work_bytes()) {
                        self.accounted_bytes = total;
                    }
                    drop(work);
                }
                return Ok(Some(index));
            }
        }
        Ok(None)
    }
}

fn checkout_work<W: WorkBytes>(
    slots: &mut WorkSlots<W>,
    key: (usize, usize, usize),
    projected_bytes: impl FnOnce() -> Result<usize, BackendError>,
    build: impl FnOnce() -> Result<W, BackendError>,
    mut evict: impl FnMut(&mut W) -> Result<bool, BackendError>,
) -> Result<(usize, W), BackendError> {
    if let Some(found) = slots.checkout_ready(key) {
        return Ok(found);
    }
    let projected_bytes = projected_bytes()?;
    loop {
        if slots.empty_index().is_some() && slots.projected_fits(projected_bytes) {
            break;
        }
        if slots.evict_idle(&mut evict)?.is_none() {
            return Err(if slots.projected_fits(projected_bytes) {
                BackendError::Unhealthy
            } else {
                BackendError::Overloaded(
                    "Qwen3.5-MoE resident Work bytes exceed the common byte budget".into(),
                )
            });
        }
    }
    let index = slots.empty_index().ok_or(BackendError::Unhealthy)?;
    slots.begin_build(index, key, projected_bytes)?;
    let work = match build() {
        Ok(work) => work,
        Err(error) => {
            slots.abort_build(index, key, projected_bytes);
            return Err(error);
        }
    };
    let actual = work.work_bytes();
    if !slots.finish_build(index, key, projected_bytes, actual) {
        slots.abort_build(index, key, projected_bytes);
        let mut work = work;
        let _ = evict(&mut work);
        drop(work);
        return Err(BackendError::Overloaded(
            "Qwen3.5-MoE actual Work bytes exceed the common byte budget".into(),
        ));
    }
    Ok((index, work))
}

fn canonicalize_selection(mut batch: Batch) -> Batch {
    let raw = batch
        .raw_indices
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut pooled_seen = std::collections::BTreeSet::new();
    batch
        .pooled_indices
        .retain(|index| !raw.contains(index) && pooled_seen.insert(*index));
    let mut raw_seen = std::collections::BTreeSet::new();
    batch.raw_indices.retain(|index| raw_seen.insert(*index));
    batch
}

enum WorkSettlement<W> {
    Cached,
    Released(W),
    Quarantined(W),
}

#[allow(clippy::too_many_arguments)]
fn settle_work<W: WorkBytes>(
    slots: &mut WorkSlots<W>,
    slot: usize,
    key: (usize, usize, usize),
    started: bool,
    proof_state: Option<bool>,
    work: W,
    mut release: impl FnMut(&mut W) -> Result<(), BackendError>,
    mut discard_unstarted: impl FnMut(&mut W) -> Result<bool, BackendError>,
) -> WorkSettlement<W> {
    if started && proof_state != Some(true) {
        return WorkSettlement::Quarantined(work);
    }
    match slots.return_ready(slot, key, work) {
        Ok(()) => WorkSettlement::Cached,
        Err(mut work) if proof_state == Some(true) => {
            if release(&mut work).is_err() {
                WorkSettlement::Quarantined(work)
            } else {
                slots.release_checked_out(slot, key);
                WorkSettlement::Released(work)
            }
        }
        Err(mut work) if !started => match discard_unstarted(&mut work) {
            Ok(true) => {
                slots.release_checked_out(slot, key);
                WorkSettlement::Released(work)
            }
            _ => WorkSettlement::Quarantined(work),
        },
        Err(work) => WorkSettlement::Quarantined(work),
    }
}

struct PendingFrame {
    lease: Option<ManuallyDrop<ExperimentalMoELease>>,
    work: Option<ManuallyDrop<ExperimentalMoEWork>>,
    engine: Option<ManuallyDrop<Arc<ExperimentalMoEEngine>>>,
    cache: Arc<Mutex<WorkSlots<ExperimentalMoEWork>>>,
    slot: usize,
    key: (usize, usize, usize),
    selected_count: usize,
    batch: Batch,
    hidden: usize,
    memory: Option<result_memory::ResultLease>,
    settled: bool,
}

impl PendingFrame {
    #[allow(clippy::too_many_arguments)]
    fn new(
        lease: ExperimentalMoELease,
        work: ExperimentalMoEWork,
        engine: Arc<ExperimentalMoEEngine>,
        cache: Arc<Mutex<WorkSlots<ExperimentalMoEWork>>>,
        slot: usize,
        key: (usize, usize, usize),
        batch: Batch,
        hidden: usize,
        memory: result_memory::ResultLease,
    ) -> Self {
        Self {
            lease: Some(ManuallyDrop::new(lease)),
            work: Some(ManuallyDrop::new(work)),
            engine: Some(ManuallyDrop::new(engine)),
            cache,
            slot,
            key,
            selected_count: 0,
            batch,
            hidden,
            memory: Some(memory),
            settled: false,
        }
    }

    fn enqueue(&mut self, pool: Pool) -> Result<(), BackendError> {
        let engine = self.engine.as_deref().ok_or(BackendError::Unhealthy)?;
        let lease = self.lease.as_deref_mut().ok_or(BackendError::Unhealthy)?;
        let work = self.work.as_deref_mut().ok_or(BackendError::Unhealthy)?;
        self.selected_count = engine
            .enqueue(work, &self.batch, lease, pool)
            .map_err(moe_error)?;
        Ok(())
    }

    fn complete(mut self) -> Result<Embeddings, BackendError> {
        let wait = match (self.engine.as_deref(), self.lease.as_deref_mut()) {
            (Some(engine), Some(lease)) => engine.wait(lease).map_err(moe_error),
            _ => Err(BackendError::Unhealthy),
        };
        if let Err(error) = wait {
            self.quarantine();
            return Err(error);
        }
        let lease = self.lease.as_deref().ok_or(BackendError::Unhealthy)?;
        let selected = lease.output(self.selected_count)?;
        let mut embeddings = output::pack(
            selected,
            self.hidden,
            self.hidden,
            &self.batch.pooled_indices,
            &self.batch.raw_indices,
            &self.batch.cumulative_seq_lengths,
        )?;
        embeddings.memory = self.memory.take();
        self.settle_success()?;
        Ok(embeddings)
    }

    fn settle_success(&mut self) -> Result<(), BackendError> {
        if self.settled {
            return Ok(());
        }
        let engine = self.engine.as_deref().ok_or(BackendError::Unhealthy)?;
        let lease = self.lease.as_deref().ok_or(BackendError::Unhealthy)?;
        let started = lease.started();
        let proof = engine.completion_proof(lease);
        let work = self
            .work
            .take()
            .map(ManuallyDrop::into_inner)
            .ok_or(BackendError::Unhealthy)?;
        let proof_state = proof.as_ref().map(|proof| work.validate_completion(proof));
        let settlement = match self.cache.lock() {
            Ok(mut cache) => settle_work(
                &mut cache,
                self.slot,
                self.key,
                started,
                proof_state,
                work,
                |work| match proof.as_ref() {
                    Some(proof) => unsafe { work.release_after_completion(proof) },
                    None => Err(BackendError::Unhealthy),
                },
                |work| unsafe { work.release_if_idle() },
            ),
            Err(_) => WorkSettlement::Quarantined(work),
        };
        match settlement {
            WorkSettlement::Cached => {}
            WorkSettlement::Released(work) => drop(work),
            WorkSettlement::Quarantined(work) => {
                std::mem::forget(work);
                self.quarantine();
                return Err(BackendError::Unhealthy);
            }
        }
        if let Some(lease) = self.lease.take() {
            drop(ManuallyDrop::into_inner(lease));
        }
        if let Some(engine) = self.engine.take() {
            drop(ManuallyDrop::into_inner(engine));
        }
        self.settled = true;
        Ok(())
    }

    fn quarantine(&mut self) {
        if let Some(engine) = self.engine.as_deref() {
            engine.poison_after_failed_fence();
        }
        if let Ok(mut slots) = self.cache.lock() {
            slots.retain_checked_out(self.slot, self.key);
        }
        if let Some(lease) = self.lease.take() {
            std::mem::forget(ManuallyDrop::into_inner(lease));
        }
        if let Some(work) = self.work.take() {
            std::mem::forget(ManuallyDrop::into_inner(work));
        }
        if let Some(engine) = self.engine.take() {
            std::mem::forget(ManuallyDrop::into_inner(engine));
        }
        self.settled = true;
    }

    fn settle_cancelled(&mut self) {
        if self.settled {
            return;
        }
        let wait = match (self.engine.as_deref(), self.lease.as_deref_mut()) {
            (Some(engine), Some(lease)) => engine.wait(lease).map_err(moe_error),
            _ => Err(BackendError::Unhealthy),
        };
        match wait {
            Ok(()) => {
                let _ = self.settle_success();
            }
            Err(_) => self.quarantine(),
        }
    }
}

impl Drop for PendingFrame {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        self.settle_cancelled();
    }
}

impl Qwen35MoeBackend {
    pub fn new(
        path: &Path,
        dtype: &str,
        model_type: ModelType,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        if !matches!(dtype, "bfloat16" | "bf16") {
            return Err(BackendError::Start(
                "Qwen3.5-MoE factory route requires bfloat16".into(),
            ));
        }
        let pool = match model_type {
            ModelType::Embedding(pool @ (Pool::Cls | Pool::Mean | Pool::LastToken)) => pool,
            ModelType::Classifier => {
                return Err(BackendError::Start(
                    "Qwen3.5-MoE factory route does not implement a score head; typed unsupported"
                        .into(),
                ))
            }
            ModelType::Decision => {
                return Err(BackendError::Start(
                    "Qwen3.5-MoE factory route does not implement Decision; typed unsupported"
                        .into(),
                ))
            }
            ModelType::Embedding(Pool::Splade) => {
                return Err(BackendError::Start(
                    "Qwen3.5-MoE factory route does not implement SPLADE; typed unsupported".into(),
                ))
            }
        };
        if !is_qwen35_moe_checkpoint(path)? {
            return Err(BackendError::Start(
                "Qwen3.5-MoE factory route requires a qwen3_5_moe checkpoint".into(),
            ));
        }
        let options = ExperimentalMoEOptions::from_environment().map_err(moe_error)?;
        let engine = ExperimentalMoEEngine::load(path, device_id, options).map_err(moe_error)?;
        Ok(Self {
            engine: Arc::new(engine),
            pool,
            work: Arc::new(Mutex::new(WorkSlots::default())),
        })
    }

    fn prepare_embedding(&self, batch: Batch) -> Result<Inference<Embeddings>, BackendError> {
        let tokens = batch.input_ids.len();
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .filter(|count| *count > 0)
            .ok_or_else(|| BackendError::Inference("missing sequence boundaries".into()))?;
        // Authoritative production validation before Work/cache mutation,
        // lease acquisition, result reservation or any launch. Runs even when
        // no pooled/raw selections are requested.
        self.engine.validate_batch(&batch).map_err(moe_error)?;
        let max_len = batch
            .cumulative_seq_lengths
            .windows(2)
            .map(|pair| pair[1].saturating_sub(pair[0]) as usize)
            .max()
            .unwrap_or(0);
        let key = (tokens, sequences, max_len);
        // Core Embeddings is last-wins and keyed by sequence index. Canonicalize
        // duplicates and raw-overrides before staging, result reservation,
        // Work selection or launch. The original request above was validated
        // exactly as supplied, including empty/media/compact cases.
        let batch = canonicalize_selection(batch);
        let raw_rows = output::raw_rows(&batch)?;
        let selected_rows = batch
            .pooled_indices
            .len()
            .checked_add(raw_rows)
            .ok_or_else(|| BackendError::Inference("selected row overflow".into()))?;
        let hidden = self.engine.hidden_size().map_err(moe_error)?;
        // Reserve canonical result memory before lease/work acquisition.
        let memory =
            result_memory::global_pool()?.reserve(output::result_bytes(&batch, hidden, hidden)?)?;
        if selected_rows == 0 {
            let mut empty = Embeddings::default();
            empty.memory = Some(memory);
            return Ok(Inference::Ready(empty));
        }
        // Reuse the common two-slot host-I/O pool as the admission primitive:
        // acquire() blocks on the condition variable and closes to Unhealthy.
        // Completion runs on the independent service completion thread.
        let lease = self.engine.lease().map_err(moe_error)?;
        let (slot, work) = {
            let mut slots = self.work.lock().map_err(|_| BackendError::Unhealthy)?;
            checkout_work(
                &mut slots,
                key,
                || {
                    self.engine
                        .work_bytes(tokens, sequences, max_len)
                        .map_err(moe_error)
                },
                || {
                    self.engine
                        .work(tokens, sequences, max_len)
                        .map_err(moe_error)
                },
                |work| unsafe { work.release_if_idle() },
            )?
        };
        // Full owner exists before Deferred is returned, so cancellation and
        // enqueue failure both traverse the same PendingFrame Drop path.
        let mut frame = PendingFrame::new(
            lease,
            work,
            Arc::clone(&self.engine),
            Arc::clone(&self.work),
            slot,
            key,
            batch,
            hidden,
            memory,
        );
        frame.enqueue(self.pool.clone())?;
        Ok(Inference::Deferred(Box::new(move || frame.complete())))
    }
}

fn hidden_width(output_len: usize, tokens: usize) -> Result<usize, BackendError> {
    output_len
        .checked_div(tokens)
        .filter(|width| *width > 0 && width.checked_mul(tokens) == Some(output_len))
        .ok_or_else(|| BackendError::Inference("Qwen3.5-MoE hidden width mismatch".into()))
}

fn moe_error(error: crate::qwen35_moe_engine::ExperimentalMoeError) -> BackendError {
    use crate::qwen35_moe_engine::ExperimentalMoeError as E;
    match error {
        E::Backend(error) => error,
        E::Cold(message) => BackendError::Start(message),
        E::MissingFa4Profile(proposal) => BackendError::Start(format!("{proposal:?}")),
        E::Admission(message) => BackendError::Inference(message),
    }
}

impl Backend for Qwen35MoeBackend {
    fn health(&self) -> Result<(), BackendError> {
        self.engine.health_check().map_err(moe_error)
    }

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        self.submit_embed(batch)?.complete()
    }

    fn submit_embed(&self, batch: Batch) -> Result<Inference<Embeddings>, BackendError> {
        self.prepare_embedding(batch)
    }

    fn predict(&self, _batch: Batch) -> Result<Predictions, BackendError> {
        Err(BackendError::Inference(
            "Qwen3.5-MoE score head is not implemented; typed unsupported".into(),
        ))
    }

    fn predict_tokens(&self, _batch: Batch) -> Result<TokenPredictions, BackendError> {
        Err(BackendError::Inference(
            "Qwen3.5-MoE token score head is not implemented; typed unsupported".into(),
        ))
    }
}

#[allow(dead_code)]
fn _owned_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::cell::Cell;
    use std::rc::Rc;

    fn fixture(value: serde_json::Value) -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "qwen35-moe-factory-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&value).unwrap()).unwrap();
        dir
    }

    #[test]
    fn factory_detects_qwen3_5_moe_before_dense_qwen35() {
        let moe = fixture(json!({
            "model_type": "qwen3_5_moe",
            "text_config": {"model_type": "qwen3_5_moe_text"}
        }));
        assert!(is_qwen35_moe_checkpoint(&moe).unwrap());
        let text = fixture(json!({"model_type": "qwen3_5_moe_text"}));
        assert!(is_qwen35_moe_checkpoint(&text).unwrap());
        let dense = fixture(json!({
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text"}
        }));
        assert!(!is_qwen35_moe_checkpoint(&dense).unwrap());
        let _ = std::fs::remove_dir_all(moe);
        let _ = std::fs::remove_dir_all(text);
        let _ = std::fs::remove_dir_all(dense);
    }
    #[derive(Clone, Debug)]
    struct FakeWork {
        live: bool,
        bytes: usize,
        releases: Rc<Cell<usize>>,
        drops: Rc<Cell<usize>>,
    }

    impl FakeWork {
        fn new(bytes: usize) -> (Self, Rc<Cell<usize>>, Rc<Cell<usize>>) {
            let releases = Rc::new(Cell::new(0));
            let drops = Rc::new(Cell::new(0));
            (
                Self {
                    live: true,
                    bytes,
                    releases: Rc::clone(&releases),
                    drops: Rc::clone(&drops),
                },
                releases,
                drops,
            )
        }

        fn release(&mut self) -> Result<(), BackendError> {
            assert!(self.live);
            self.live = false;
            self.releases.set(self.releases.get() + 1);
            Ok(())
        }

        fn discard(&mut self) -> Result<bool, BackendError> {
            self.live = false;
            Ok(true)
        }
    }

    impl WorkBytes for FakeWork {
        fn work_bytes(&self) -> usize {
            self.bytes
        }
    }

    impl Drop for FakeWork {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    fn test_batch(pooled: Vec<u32>, raw: Vec<u32>) -> Batch {
        Batch {
            multimodal: Vec::new(),
            input_ids: (20..27).collect(),
            token_type_ids: vec![0; 7],
            position_ids: (0..7).collect(),
            cumulative_seq_lengths: vec![0, 7],
            max_length: 7,
            pooled_indices: pooled,
            raw_indices: raw,
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn validate_test_batch(batch: &Batch) -> Result<usize, BackendError> {
        crate::validate_qwen35_batch_geometry(batch, 262144, 300000, 2048, 512, 512, 2048)
    }

    #[test]
    fn workslots_reuse_returns_and_accounts_bytes() {
        let mut slots: WorkSlots<FakeWork> = WorkSlots::default();
        assert_eq!(slots.byte_budget, crate::cache::BYTE_BUDGET);
        let (first, _, _) = FakeWork::new(64);
        let (second, _, _) = FakeWork::new(32);
        slots.install_ready(0, (7, 1, 7), first).unwrap();
        slots.install_ready(1, (7, 1, 7), second).unwrap();
        assert_eq!(slots.resident_bytes(), 96);
        let (index, work) = slots.checkout_ready((7, 1, 7)).unwrap();
        assert_eq!(index, 0);
        assert_eq!(slots.resident_bytes(), 96);
        slots.return_ready(index, (7, 1, 7), work).unwrap();
        let (index, work) = slots.checkout_ready((7, 1, 7)).unwrap();
        slots.release_checked_out(index, (7, 1, 7));
        assert_eq!(slots.resident_bytes(), 32);
        drop(work);
    }

    #[test]
    fn different_key_b_can_use_free_slot_while_a_is_checked_out() {
        let key_a = (7, 1, 7);
        let key_b = (8, 1, 8);
        let mut slots: WorkSlots<FakeWork> = WorkSlots::default();
        let (a, _, _) = FakeWork::new(70);
        let (b, _, _) = FakeWork::new(80);
        slots.install_ready(0, key_a, a).unwrap();
        let (a_index, a_work) = slots.checkout_ready(key_a).unwrap();
        slots.install_ready(1, key_b, b).unwrap();
        let (b_index, b_work) = slots.checkout_ready(key_b).unwrap();
        assert_ne!(a_index, b_index);
        assert_eq!(slots.resident_bytes(), 150);
        slots.return_ready(b_index, key_b, b_work).unwrap();
        slots.return_ready(a_index, key_a, a_work).unwrap();
        assert_eq!(slots.resident_bytes(), 150);
    }

    #[test]
    fn constructor_failure_retains_a_and_rolls_back_only_idle_slot() {
        let key_a = (7, 1, 7);
        let key_b = (8, 1, 8);
        let key_c = (9, 1, 9);
        let mut slots: WorkSlots<FakeWork> = WorkSlots::default();
        let (a, _, _) = FakeWork::new(70);
        let (idle, _, idle_drops) = FakeWork::new(15);
        slots.install_ready(0, key_a, a).unwrap();
        let (a_index, a_work) = slots.checkout_ready(key_a).unwrap();
        slots.install_ready(1, key_c, idle).unwrap();
        let error = checkout_work(
            &mut slots,
            key_b,
            || Ok(80),
            || Err(BackendError::Unhealthy),
            FakeWork::discard,
        )
        .unwrap_err();
        assert!(matches!(error, BackendError::Unhealthy));
        assert_eq!(idle_drops.get(), 1);
        assert_eq!(slots.resident_bytes(), 70);
        slots.return_ready(a_index, key_a, a_work).unwrap();
        assert_eq!(slots.resident_bytes(), 70);
    }

    #[test]
    fn cached_ready_hit_never_invokes_plan_or_build() {
        let key = (7, 1, 7);
        let mut slots: WorkSlots<FakeWork> = WorkSlots::default();
        let (work, _, _) = FakeWork::new(64);
        slots.install_ready(0, key, work).unwrap();
        let (index, work) = checkout_work(
            &mut slots,
            key,
            || panic!("warm cache hit must not plan Work bytes"),
            || panic!("warm cache hit must not build a Work"),
            |_: &mut FakeWork| -> Result<bool, BackendError> {
                panic!("warm cache hit must not evict")
            },
        )
        .unwrap();
        assert_eq!(index, 0);
        assert!(work.live);
    }

    #[test]
    fn cache_miss_plans_once_and_overbudget_rejects_before_build() {
        let mut slots: WorkSlots<FakeWork> = WorkSlots::with_budget(10);
        let (held, _, _) = FakeWork::new(5);
        slots.install_ready(0, (6, 1, 6), held).unwrap();
        let (_held_index, held_work) = slots.checkout_ready((6, 1, 6)).unwrap();
        let plans = Rc::new(Cell::new(0usize));
        let builds = Rc::new(Cell::new(0usize));
        let plan_count = Rc::clone(&plans);
        let build_count = Rc::clone(&builds);
        let error = checkout_work(
            &mut slots,
            (7, 1, 7),
            || {
                plan_count.set(plan_count.get() + 1);
                Ok(20)
            },
            || {
                build_count.set(build_count.get() + 1);
                Ok(FakeWork::new(20).0)
            },
            FakeWork::discard,
        )
        .unwrap_err();
        assert!(matches!(error, BackendError::Overloaded(_)));
        assert_eq!(plans.get(), 1);
        assert_eq!(builds.get(), 0);
        assert_eq!(slots.resident_bytes(), 5);
        drop(held_work);
    }

    #[test]
    fn work_budget_admits_under_budget_and_rejects_before_allocation() {
        let key_a = (7, 1, 7);
        let key_b = (8, 1, 8);
        let key_c = (9, 1, 9);
        let mut slots: WorkSlots<FakeWork> = WorkSlots::with_budget(100);
        let (a, _, _) = FakeWork::new(60);
        slots.install_ready(0, key_a, a).unwrap();
        let (a_index, a_work) = slots.checkout_ready(key_a).unwrap();
        let mut builds = 0usize;
        let (b_index, b_work) = checkout_work(
            &mut slots,
            key_b,
            || Ok(30),
            || {
                builds += 1;
                Ok(FakeWork::new(30).0)
            },
            FakeWork::discard,
        )
        .unwrap();
        assert_eq!(builds, 1);
        assert_eq!(slots.resident_bytes(), 90);
        let error = checkout_work(
            &mut slots,
            key_c,
            || Ok(20),
            || {
                builds += 1;
                Ok(FakeWork::new(20).0)
            },
            FakeWork::discard,
        )
        .unwrap_err();
        assert!(matches!(error, BackendError::Overloaded(_)));
        assert_eq!(builds, 1);
        assert_eq!(slots.resident_bytes(), 90);
        slots.release_checked_out(b_index, key_b);
        slots.release_checked_out(a_index, key_a);
        drop(b_work);
        drop(a_work);
    }

    #[test]
    fn one_oversized_active_is_allowed_but_blocks_second_resident_and_retained_counts() {
        let key_a = (7, 1, 7);
        let key_b = (8, 1, 8);
        let mut slots: WorkSlots<FakeWork> = WorkSlots::with_budget(100);
        let (a_index, a_work) = checkout_work(
            &mut slots,
            key_a,
            || Ok(150),
            || Ok(FakeWork::new(150).0),
            FakeWork::discard,
        )
        .unwrap();
        assert_eq!(slots.resident_bytes(), 150);
        let mut builds = 0usize;
        let error = checkout_work(
            &mut slots,
            key_b,
            || Ok(1),
            || {
                builds += 1;
                Ok(FakeWork::new(1).0)
            },
            FakeWork::discard,
        )
        .unwrap_err();
        assert!(matches!(error, BackendError::Overloaded(_)));
        assert_eq!(builds, 0);
        slots.retain_checked_out(a_index, key_a);
        assert_eq!(slots.resident_bytes(), 150);
        let error = checkout_work(
            &mut slots,
            key_b,
            || Ok(1),
            || {
                builds += 1;
                Ok(FakeWork::new(1).0)
            },
            FakeWork::discard,
        )
        .unwrap_err();
        assert!(matches!(error, BackendError::Overloaded(_)));
        assert_eq!(builds, 0);
        std::mem::forget(a_work);
    }

    #[test]
    fn budget_evicts_only_ready_idle_and_preserves_checked_out_a() {
        let key_a = (7, 1, 7);
        let key_b = (8, 1, 8);
        let key_c = (9, 1, 9);
        let mut slots: WorkSlots<FakeWork> = WorkSlots::with_budget(100);
        let (a, _, _) = FakeWork::new(40);
        let (idle, _, idle_drops) = FakeWork::new(30);
        slots.install_ready(0, key_a, a).unwrap();
        let (a_index, a_work) = slots.checkout_ready(key_a).unwrap();
        slots.install_ready(1, key_c, idle).unwrap();
        let (b_index, b_work) = checkout_work(
            &mut slots,
            key_b,
            || Ok(50),
            || Ok(FakeWork::new(50).0),
            FakeWork::discard,
        )
        .unwrap();
        assert_eq!(idle_drops.get(), 1);
        assert_eq!(slots.resident_bytes(), 90);
        slots.return_ready(b_index, key_b, b_work).unwrap();
        slots.return_ready(a_index, key_a, a_work).unwrap();
    }

    #[test]
    fn settle_work_caches_live_work_and_releases_only_on_discard() {
        let key = (7, 1, 7);
        let mut slots: WorkSlots<FakeWork> = WorkSlots::default();
        let (first, releases, _) = FakeWork::new(64);
        let (second, _, _) = FakeWork::new(64);
        slots.install_ready(0, key, first).unwrap();
        slots.install_ready(1, key, second).unwrap();
        let (slot, work) = slots.checkout_ready(key).unwrap();
        let outcome = settle_work(
            &mut slots,
            slot,
            key,
            true,
            Some(true),
            work,
            FakeWork::release,
            FakeWork::discard,
        );
        assert!(matches!(outcome, WorkSettlement::Cached));
        assert_eq!(releases.get(), 0);
        let (reused_index, reused) = slots.checkout_ready(key).unwrap();
        assert!(reused.live);
        slots.release_checked_out(reused_index, key);
        drop(reused);

        let (discard, discard_releases, _) = FakeWork::new(64);
        slots.install_ready(0, key, discard).unwrap();
        let (slot, work) = slots.checkout_ready(key).unwrap();
        let outcome = settle_work(
            &mut slots,
            slot,
            (8, 1, 8),
            true,
            Some(true),
            work,
            FakeWork::release,
            FakeWork::discard,
        );
        assert!(matches!(outcome, WorkSettlement::Released(_)));
        assert_eq!(discard_releases.get(), 1);
    }

    #[test]
    fn settle_work_quarantines_started_unproven_and_mismatch() {
        let key = (7, 1, 7);
        for (proof, started) in [(None, true), (Some(false), true)] {
            let mut slots: WorkSlots<FakeWork> = WorkSlots::default();
            let (work, releases, drops) = FakeWork::new(64);
            slots.install_ready(0, key, work).unwrap();
            let (slot, work) = slots.checkout_ready(key).unwrap();
            let outcome = settle_work(
                &mut slots,
                slot,
                key,
                started,
                proof,
                work,
                FakeWork::release,
                FakeWork::discard,
            );
            let work = match outcome {
                WorkSettlement::Quarantined(work) => work,
                _ => panic!("expected quarantine"),
            };
            assert!(work.live);
            assert_eq!(releases.get(), 0);
            std::mem::forget(work);
            assert_eq!(drops.get(), 0);
        }
    }

    #[test]
    fn never_queued_work_returns_to_its_exact_slot() {
        let key = (7, 1, 7);
        let mut slots: WorkSlots<FakeWork> = WorkSlots::default();
        let (work, releases, _) = FakeWork::new(64);
        slots.install_ready(0, key, work).unwrap();
        let (slot, work) = slots.checkout_ready(key).unwrap();
        let outcome = settle_work(
            &mut slots,
            slot,
            key,
            false,
            None,
            work,
            FakeWork::release,
            FakeWork::discard,
        );
        assert!(matches!(outcome, WorkSettlement::Cached));
        assert_eq!(releases.get(), 0);
        let (_, reused) = slots.checkout_ready(key).unwrap();
        assert!(reused.live);
    }

    struct TestSlot(u32);
    impl crate::staging::SlotBytes for TestSlot {
        fn slot_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn common_pool_blocks_third_and_wakes_after_one_release() {
        use std::sync::mpsc;
        use std::time::Duration;
        let pool = crate::staging::SlotPool::new([TestSlot(1), TestSlot(2)]);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        let (started, waiting) = mpsc::channel();
        let (result, receiver) = mpsc::channel();
        let waiter = {
            let pool = pool.clone();
            std::thread::spawn(move || {
                started.send(()).unwrap();
                result.send(pool.acquire()).unwrap();
            })
        };
        waiting.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(receiver.recv_timeout(Duration::from_millis(30)).is_err());
        pool.release(second);
        assert!(wait_for_acquire(&receiver).is_ok());
        waiter.join().unwrap();
        pool.release(first);
    }

    fn wait_for_acquire(
        receiver: &std::sync::mpsc::Receiver<Result<TestSlot, BackendError>>,
    ) -> Result<TestSlot, BackendError> {
        receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("third acquire did not wake")
    }

    #[test]
    fn common_pool_close_wakes_waiter_typed_unhealthy() {
        use std::sync::mpsc;
        use std::time::Duration;
        let pool = crate::staging::SlotPool::new([TestSlot(1), TestSlot(2)]);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        let (started, waiting) = mpsc::channel();
        let (result, receiver) = mpsc::channel();
        let waiter = {
            let pool = pool.clone();
            std::thread::spawn(move || {
                started.send(()).unwrap();
                result.send(pool.acquire()).unwrap();
            })
        };
        waiting.recv_timeout(Duration::from_secs(2)).unwrap();
        std::thread::sleep(Duration::from_millis(30));
        pool.close();
        assert!(matches!(
            wait_for_acquire(&receiver),
            Err(BackendError::Unhealthy)
        ));
        waiter.join().unwrap();
        pool.release(first);
        pool.release(second);
    }

    #[test]
    fn duplicate_canonicalization_preserves_last_wins_without_hard_capacity_rejection() {
        let duplicate_pooled = test_batch(vec![0, 0], Vec::new());
        assert!(validate_test_batch(&duplicate_pooled).is_ok());
        let canonical = canonicalize_selection(duplicate_pooled);
        assert_eq!(canonical.pooled_indices, vec![0]);
        assert!(canonical.raw_indices.is_empty());

        let duplicate_raw = test_batch(Vec::new(), vec![0, 0]);
        assert!(validate_test_batch(&duplicate_raw).is_ok());
        let canonical = canonicalize_selection(duplicate_raw);
        assert_eq!(canonical.raw_indices, vec![0]);

        let overridden = test_batch(vec![0], vec![0]);
        assert!(validate_test_batch(&overridden).is_ok());
        let canonical = canonicalize_selection(overridden);
        assert!(canonical.pooled_indices.is_empty());
        assert_eq!(canonical.raw_indices, vec![0]);
    }

    fn embedding_bits(
        result: &Embeddings,
    ) -> std::collections::BTreeMap<usize, (String, Vec<u32>)> {
        result
            .iter()
            .map(|(index, embedding)| {
                let (kind, values) = match embedding {
                    infinity_v3_backend_core::Embedding::Pooled(values) => {
                        ("pooled".to_string(), values.clone())
                    }
                    infinity_v3_backend_core::Embedding::All(rows) => (
                        "raw".to_string(),
                        rows.iter().flat_map(|row| row.iter().copied()).collect(),
                    ),
                };
                (
                    *index,
                    (kind, values.into_iter().map(f32::to_bits).collect()),
                )
            })
            .collect()
    }

    fn pack_for(batch: &Batch, selected: &[f32]) -> Embeddings {
        output::pack(
            selected,
            2,
            2,
            &batch.pooled_indices,
            &batch.raw_indices,
            &batch.cumulative_seq_lengths,
        )
        .unwrap()
    }

    #[test]
    fn duplicate_pooled_canonicalization_preserves_original_embeddings_map() {
        let original = test_batch(vec![0, 0], Vec::new());
        let canonical = canonicalize_selection(original.clone());
        let original_selected = [1.0f32, 2.0, 9.0, 10.0];
        let canonical_selected = [9.0f32, 10.0];
        assert_eq!(
            embedding_bits(&pack_for(&original, &original_selected)),
            embedding_bits(&pack_for(&canonical, &canonical_selected))
        );
    }

    #[test]
    fn duplicate_raw_canonicalization_preserves_original_embeddings_map() {
        let original = test_batch(Vec::new(), vec![0, 0]);
        let canonical = canonicalize_selection(original.clone());
        let first = (0..14).map(|value| value as f32).collect::<Vec<_>>();
        let second = (100..114).map(|value| value as f32).collect::<Vec<_>>();
        let mut original_selected = first;
        original_selected.extend_from_slice(&second);
        assert_eq!(
            embedding_bits(&pack_for(&original, &original_selected)),
            embedding_bits(&pack_for(&canonical, &second))
        );
    }

    #[test]
    fn raw_override_canonicalization_preserves_original_embeddings_map() {
        let original = test_batch(vec![0], vec![0]);
        let canonical = canonicalize_selection(original.clone());
        let mut original_selected = vec![1.0f32, 2.0];
        let raw = (10..24).map(|value| value as f32).collect::<Vec<_>>();
        original_selected.extend_from_slice(&raw);
        assert_eq!(
            embedding_bits(&pack_for(&original, &original_selected)),
            embedding_bits(&pack_for(&canonical, &raw))
        );
    }

    #[test]
    fn original_invalid_requests_are_validated_before_canonicalization() {
        use std::sync::Arc as StdArc;

        let mut media = test_batch(vec![0, 0], vec![0, 0]);
        media.multimodal = vec![Some(StdArc::new(
            infinity_v3_backend_core::MultimodalEncoding {
                images: Vec::new(),
                audios: Vec::new(),
                position_ids: [Vec::new(), Vec::new(), Vec::new()],
                memory: None,
            },
        ))];
        assert!(validate_test_batch(&media).is_err());

        let mut compact = test_batch(vec![0, 0], vec![0, 0]);
        compact.compact_input_ids = Some(compact.input_ids.clone());
        assert!(validate_test_batch(&compact).is_err());

        let empty = Batch {
            input_ids: Vec::new(),
            token_type_ids: Vec::new(),
            position_ids: Vec::new(),
            cumulative_seq_lengths: vec![0, 0],
            max_length: 0,
            ..test_batch(Vec::new(), Vec::new())
        };
        assert!(validate_test_batch(&empty).is_err());
    }
    #[test]
    fn factory_error_classification_preserves_backend_overloaded_and_unhealthy() {
        use crate::qwen35_moe_engine::ExperimentalMoeError as E;
        assert!(matches!(
            moe_error(E::Backend(BackendError::Overloaded("busy".into()))),
            BackendError::Overloaded(_)
        ));
        assert!(matches!(
            moe_error(E::Backend(BackendError::Unhealthy)),
            BackendError::Unhealthy
        ));
        assert!(matches!(
            moe_error(E::Cold("cold".into())),
            BackendError::Start(message) if message == "cold"
        ));
        assert!(matches!(
            moe_error(E::Admission("runtime".into())),
            BackendError::Inference(message) if message == "runtime"
        ));
    }
}
