use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend::{Backend, BackendKind, DType as FactoryDType, ModelType, TorchDevice};
use infinity_v3_backend_core::{
    Backend as CoreBackend, BackendError, Batch, Embedding, Embeddings, Predictions,
    TokenPredictions,
};

use crate::report::{
    CaseOutput, EmbeddingEntry, EmbeddingResult, PredictionResult, PredictionRow, StructuredError,
    TokenPredictionResult, TokenPredictionSequence, TokenRow,
};
use crate::schema::{ExpectedCanonical, ExpectedKind, PoolArg, Task};
use crate::ProbeError;

pub enum Engine {
    Factory(Box<Backend>),
    Core(Box<OxideBackend>),
}

pub enum Held {
    Embed(infinity_v3_backend_core::Inference<Embeddings>),
    Predict(infinity_v3_backend_core::Inference<Predictions>),
}

pub enum Completed {
    Embed(Embeddings),
    Predict(Predictions),
}

impl Held {
    pub fn complete(self) -> Result<Completed, BackendError> {
        match self {
            Self::Embed(inference) => inference.complete().map(Completed::Embed),
            Self::Predict(inference) => inference.complete().map(Completed::Predict),
        }
    }

    pub fn is_deferred(&self) -> bool {
        match self {
            Self::Embed(inference) => {
                matches!(inference, infinity_v3_backend_core::Inference::Deferred(_))
            }
            Self::Predict(inference) => {
                matches!(inference, infinity_v3_backend_core::Inference::Deferred(_))
            }
        }
    }
}

#[derive(Debug)]
pub struct LeaseTracker {
    capacity: usize,
    live: AtomicUsize,
    max_observed: AtomicUsize,
}

impl LeaseTracker {
    pub fn new(capacity: usize) -> Arc<Self> {
        assert!(capacity > 0, "lease capacity must be positive");
        Arc::new(Self {
            capacity,
            live: AtomicUsize::new(0),
            max_observed: AtomicUsize::new(0),
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn live(&self) -> usize {
        self.live.load(Ordering::Acquire)
    }

    pub fn max_observed(&self) -> usize {
        self.max_observed.load(Ordering::Acquire)
    }

    fn reserve(self: &Arc<Self>) -> Result<SlotReservation, BackendError> {
        let previous = self
            .live
            .try_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                if live < self.capacity {
                    Some(live + 1)
                } else {
                    None
                }
            })
            .map_err(|live| {
                BackendError::Overloaded(format!(
                    "probe lease capacity {} reached with {live} live leases",
                    self.capacity
                ))
            })?;
        self.max_observed.fetch_max(previous + 1, Ordering::AcqRel);
        Ok(SlotReservation {
            tracker: self.clone(),
            committed: false,
        })
    }

    fn release(&self) {
        let previous = self.live.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "lease tracker underflow");
    }
}

#[derive(Debug)]
pub struct SlotReservation {
    tracker: Arc<LeaseTracker>,
    committed: bool,
}

impl SlotReservation {
    pub fn commit(mut self, held: Held) -> HeldLease {
        self.committed = true;
        HeldLease {
            held: Some(held),
            tracker: self.tracker.clone(),
        }
    }
}

impl Drop for SlotReservation {
    fn drop(&mut self) {
        if !self.committed {
            self.tracker.release();
        }
    }
}

pub struct HeldLease {
    held: Option<Held>,
    tracker: Arc<LeaseTracker>,
}

impl HeldLease {
    pub fn is_deferred(&self) -> bool {
        self.held.as_ref().is_some_and(Held::is_deferred)
    }

    pub fn complete(mut self) -> Result<Completed, BackendError> {
        let held = self
            .held
            .take()
            .expect("held lease must own one inference handle");
        let result = held.complete();
        self.tracker.release();
        result
    }
}

impl Drop for HeldLease {
    fn drop(&mut self) {
        if self.held.is_some() {
            self.tracker.release();
        }
    }
}

pub struct TrackedSubmit;

impl TrackedSubmit {
    pub fn submit(
        backend: &OxideBackend,
        task: Task,
        batch: Batch,
        tracker: &Arc<LeaseTracker>,
    ) -> Result<HeldLease, ProbeError> {
        let reservation = tracker
            .reserve()
            .map_err(|err| ProbeError::Backend(backend_error("lifecycle_reserve", &err)))?;
        let held = submit_held(backend, task, batch)
            .map_err(|err| ProbeError::Backend(backend_error("lifecycle_submit", &err)))?;
        Ok(reservation.commit(held))
    }
}

pub fn submit_held(backend: &OxideBackend, task: Task, batch: Batch) -> Result<Held, BackendError> {
    match task {
        Task::Embedding => CoreBackend::submit_embed(backend, batch).map(Held::Embed),
        Task::Classifier => CoreBackend::submit_predict(backend, batch).map(Held::Predict),
    }
}

pub async fn construct_factory(
    model_path: &std::path::Path,
    dtype: FactoryDType,
    model_type: ModelType,
) -> Result<Engine, BackendError> {
    Backend::new_selected(
        model_path.to_path_buf(),
        None,
        dtype,
        model_type,
        None,
        0,
        false,
        BackendKind::Oxide,
        TorchDevice::Cuda,
    )
    .await
    .map(|backend| Engine::Factory(Box::new(backend)))
}

pub fn construct_core(
    model_path: &std::path::Path,
    model_type: ModelType,
) -> Result<Engine, BackendError> {
    OxideBackend::new(model_path, "bfloat16", model_type, 0)
        .map(|backend| Engine::Core(Box::new(backend)))
}

pub fn model_type(task: Task, pool: PoolArg) -> ModelType {
    match task {
        Task::Embedding => ModelType::Embedding(pool.backend_pool()),
        Task::Classifier => ModelType::Classifier,
    }
}

impl Engine {
    pub async fn embed(
        &self,
        batch: Batch,
    ) -> Result<(Embeddings, Option<Duration>), BackendError> {
        match self {
            Self::Factory(backend) => {
                let (result, duration) = backend.embed(batch).await?;
                Ok((result, Some(duration)))
            }
            Self::Core(backend) => Ok((CoreBackend::embed(backend.as_ref(), batch)?, None)),
        }
    }

    pub async fn predict(
        &self,
        batch: Batch,
    ) -> Result<(Predictions, Option<Duration>), BackendError> {
        match self {
            Self::Factory(backend) => {
                let (result, duration) = backend.predict(batch).await?;
                Ok((result, Some(duration)))
            }
            Self::Core(backend) => Ok((CoreBackend::predict(backend.as_ref(), batch)?, None)),
        }
    }

    pub async fn predict_tokens(
        &self,
        batch: Batch,
    ) -> Result<(TokenPredictions, Option<Duration>), BackendError> {
        match self {
            Self::Factory(backend) => {
                let (result, duration) = backend.predict_tokens(batch).await?;
                Ok((result, Some(duration)))
            }
            Self::Core(backend) => {
                Ok((CoreBackend::predict_tokens(backend.as_ref(), batch)?, None))
            }
        }
    }

    pub async fn health(&self) -> Result<(), BackendError> {
        match self {
            Self::Factory(backend) => backend.health().await,
            Self::Core(backend) => CoreBackend::health(backend.as_ref()),
        }
    }

    pub fn core(&self) -> Option<&OxideBackend> {
        match self {
            Self::Factory(_) => None,
            Self::Core(backend) => Some(backend.as_ref()),
        }
    }
}

pub fn backend_error(kind_phase: &str, error: &BackendError) -> StructuredError {
    let (kind, message) = match error {
        BackendError::Overloaded(message) => ("overloaded", message.clone()),
        BackendError::NoBackend => ("no_backend", "No backend found".into()),
        BackendError::Start(message) => ("start", message.clone()),
        BackendError::Inference(message) => ("inference", message.clone()),
        BackendError::Unhealthy => ("unhealthy", "Backend is unhealthy".into()),
        BackendError::WeightsNotFound(message) => ("weights_not_found", message.clone()),
    };
    StructuredError::new(kind, kind_phase, message)
}

fn finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

pub fn embedding_output(
    expected: &ExpectedCanonical,
    embeddings: &Embeddings,
) -> Result<CaseOutput, ProbeError> {
    if embeddings.len() != expected.entries.len() {
        return Err(ProbeError::Output(format!(
            "backend returned {} embedding entries, expected {}",
            embeddings.len(),
            expected.entries.len()
        )));
    }
    let mut entries = Vec::with_capacity(expected.entries.len());
    let mut width: Option<usize> = None;
    for (&index, wanted) in &expected.entries {
        let actual = embeddings.get(&index).ok_or_else(|| {
            ProbeError::Output(format!("backend output is missing expected index {index}"))
        })?;
        match (wanted.kind, actual) {
            (ExpectedKind::Pooled, Embedding::Pooled(values)) => {
                if !finite(values) {
                    return Err(ProbeError::Output(format!(
                        "pooled index {index} contains a non-finite value"
                    )));
                }
                if let Some(width) = width {
                    if values.len() != width {
                        return Err(ProbeError::Output(format!(
                            "pooled index {index} width {} differs from {width}",
                            values.len()
                        )));
                    }
                } else {
                    width = Some(values.len());
                }
                entries.push(EmbeddingEntry {
                    index,
                    kind: "pooled".into(),
                    shape: vec![values.len()],
                    values: values.clone(),
                    bits: values.iter().map(|value| value.to_bits()).collect(),
                    finite: true,
                });
            }
            (ExpectedKind::Raw, Embedding::All(rows)) => {
                if rows.len() != wanted.rows {
                    return Err(ProbeError::Output(format!(
                        "raw index {index} has {} rows, expected {}",
                        rows.len(),
                        wanted.rows
                    )));
                }
                for (row_index, row) in rows.iter().enumerate() {
                    if !finite(row) {
                        return Err(ProbeError::Output(format!(
                            "raw index {index} row {row_index} contains a non-finite value"
                        )));
                    }
                    if let Some(width) = width {
                        if row.len() != width {
                            return Err(ProbeError::Output(format!(
                                "raw index {index} row {row_index} width {} differs from {width}",
                                row.len()
                            )));
                        }
                    } else {
                        width = Some(row.len());
                    }
                }
                let mut values = Vec::new();
                for row in rows {
                    values.extend_from_slice(row);
                }
                entries.push(EmbeddingEntry {
                    index,
                    kind: "raw".into(),
                    shape: vec![rows.len(), width.unwrap_or(0)],
                    values: values.clone(),
                    bits: values.iter().map(|value| value.to_bits()).collect(),
                    finite: true,
                });
            }
            (ExpectedKind::Pooled, Embedding::All(_)) => {
                return Err(ProbeError::Output(format!(
                    "index {index} is expected to be pooled but backend returned All"
                )));
            }
            (ExpectedKind::Raw, Embedding::Pooled(_)) => {
                return Err(ProbeError::Output(format!(
                    "index {index} is expected to be raw but backend returned Pooled"
                )));
            }
        }
    }
    Ok(CaseOutput::Embedding(EmbeddingResult {
        api_value_kind: "embeddings".into(),
        entry_count: entries.len(),
        order: "ascending_index".into(),
        raw_overrides_pooled: expected.raw_overrides_pooled.clone(),
        duplicate_selection_indices: expected.duplicate_selection_indices.clone(),
        entries,
    }))
}

pub fn require_class_width(output: &CaseOutput, expected: usize) -> Result<(), ProbeError> {
    let actual = match output {
        CaseOutput::Predictions(value) => value.class_count,
        CaseOutput::TokenPredictions(value) => value.class_count,
        CaseOutput::Embedding(_) => {
            return Err(ProbeError::Output(
                "classifier returned an embedding result".into(),
            ))
        }
    };
    if expected == 0 || actual != expected {
        return Err(ProbeError::Output(format!(
            "classifier returned {actual} labels, checkpoint declares {expected}"
        )));
    }
    Ok(())
}

pub fn predictions_output(
    predictions: &Predictions,
    sequences: usize,
) -> Result<CaseOutput, ProbeError> {
    if predictions.len() != sequences {
        return Err(ProbeError::Output(format!(
            "backend returned {} prediction rows, expected {sequences}",
            predictions.len()
        )));
    }
    let mut rows = Vec::with_capacity(sequences);
    let mut class_count: Option<usize> = None;
    for index in 0..sequences {
        let values = predictions.get(&index).ok_or_else(|| {
            ProbeError::Output(format!("backend output is missing canonical row {index}"))
        })?;
        if !finite(values) {
            return Err(ProbeError::Output(format!(
                "prediction row {index} contains a non-finite value"
            )));
        }
        if let Some(width) = class_count {
            if values.len() != width {
                return Err(ProbeError::Output(format!(
                    "prediction row {index} width {} differs from {width}",
                    values.len()
                )));
            }
        } else {
            class_count = Some(values.len());
        }
        rows.push(PredictionRow {
            index,
            values: values.clone(),
            bits: values.iter().map(|value| value.to_bits()).collect(),
            finite: true,
        });
    }
    Ok(CaseOutput::Predictions(PredictionResult {
        api_value_kind: "predictions".into(),
        row_count: rows.len(),
        class_count: class_count.unwrap_or(0),
        rows,
    }))
}

pub fn token_predictions_output(
    predictions: &TokenPredictions,
    sequence_lengths: &[usize],
) -> Result<CaseOutput, ProbeError> {
    if predictions.len() != sequence_lengths.len() {
        return Err(ProbeError::Output(format!(
            "backend returned {} token prediction sequences, expected {}",
            predictions.len(),
            sequence_lengths.len()
        )));
    }
    let mut sequences = Vec::with_capacity(sequence_lengths.len());
    let mut class_count: Option<usize> = None;
    for (index, &token_count) in sequence_lengths.iter().enumerate() {
        let rows = predictions.get(&index).ok_or_else(|| {
            ProbeError::Output(format!(
                "backend output is missing token prediction sequence {index}"
            ))
        })?;
        if rows.len() != token_count {
            return Err(ProbeError::Output(format!(
                "token prediction sequence {index} has {} rows, expected {token_count}",
                rows.len()
            )));
        }
        let mut token_rows = Vec::with_capacity(rows.len());
        for (token_index, row) in rows.iter().enumerate() {
            if !finite(row) {
                return Err(ProbeError::Output(format!(
                    "token prediction sequence {index} token {token_index} contains a non-finite value"
                )));
            }
            if let Some(width) = class_count {
                if row.len() != width {
                    return Err(ProbeError::Output(format!(
                        "token prediction sequence {index} token {token_index} width {} differs from {width}",
                        row.len()
                    )));
                }
            } else {
                class_count = Some(row.len());
            }
            token_rows.push(TokenRow {
                token_index,
                values: row.clone(),
                bits: row.iter().map(|value| value.to_bits()).collect(),
                finite: true,
            });
        }
        sequences.push(TokenPredictionSequence {
            index,
            token_count: token_rows.len(),
            rows: token_rows,
        });
    }
    Ok(CaseOutput::TokenPredictions(TokenPredictionResult {
        api_value_kind: "token_predictions".into(),
        sequence_count: sequences.len(),
        class_count: class_count.unwrap_or(0),
        sequences,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_width_is_checked_against_checkpoint_not_only_other_rows() {
        let mut predictions = Predictions::default();
        predictions.insert(0, vec![1.0, 2.0]);
        let output = predictions_output(&predictions, 1).unwrap();
        assert!(require_class_width(&output, 2).is_ok());
        assert!(require_class_width(&output, 1).is_err());
        assert!(require_class_width(&output, 0).is_err());
    }

    #[test]
    fn tracker_caps_two_live_leases_and_releases_on_drop() {
        let tracker = LeaseTracker::new(2);
        let a = tracker.reserve().expect("first");
        let b = tracker.reserve().expect("second");
        assert_eq!(tracker.live(), 2);
        assert_eq!(tracker.max_observed(), 2);
        assert!(
            tracker.reserve().is_err(),
            "third slot must not be acquired"
        );
        drop(a);
        assert_eq!(tracker.live(), 1);
        drop(b);
        assert_eq!(tracker.live(), 0);
    }

    #[test]
    fn failed_submission_releases_reserved_slot() {
        let tracker = LeaseTracker::new(2);
        {
            let reservation = tracker.reserve().expect("reserve");
            assert_eq!(tracker.live(), 1);
            drop(reservation);
        }
        assert_eq!(tracker.live(), 0);
        assert_eq!(tracker.max_observed(), 1);
    }
}
