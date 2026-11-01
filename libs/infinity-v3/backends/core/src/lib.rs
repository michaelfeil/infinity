pub mod block_attention_config;
mod multimodal;
pub mod result_memory;
#[cfg(feature = "clap")]
use clap::ValueEnum;
pub use multimodal::{AudioFeatures, ImagePatches, MultimodalEncoding};
use serde::Deserialize;
use std::fmt;
use std::ops::Deref;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct Batch {
    /// Empty for text-only batches, otherwise one entry per sequence.
    pub multimodal: Vec<Option<std::sync::Arc<MultimodalEncoding>>>,
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub cumulative_seq_lengths: Vec<u32>,
    pub max_length: u32,
    pub pooled_indices: Vec<u32>,
    pub raw_indices: Vec<u32>,
    pub compact_input_ids: Option<Vec<u32>>,
    pub compact_position_ids: Option<Vec<u32>>,
    pub scatter_unfold: Option<Vec<u32>>,
    pub fold_gather: Option<Vec<u32>>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.cumulative_seq_lengths.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Float data that is either owned by Rust or shared from an external owner.
#[derive(Clone)]
pub struct SharedF32(Storage);

#[derive(Clone)]
enum Storage {
    Owned(Vec<f32>),
    Shared {
        bytes: bytes::Bytes,
        offset: usize,
        len: usize,
    },
}

impl Storage {
    fn owned_mut(&mut self) -> &mut Vec<f32> {
        loop {
            match self {
                Self::Owned(values) => return values,
                Self::Shared { bytes, offset, len } => {
                    let values = shared_slice(bytes, *offset, *len).to_vec();
                    *self = Self::Owned(values);
                }
            }
        }
    }
}

fn shared_slice(bytes: &bytes::Bytes, offset: usize, len: usize) -> &[f32] {
    // SAFETY: the private storage can only be created by `shared`, which checks
    // the complete byte range and actual pointer alignment. Truncation only
    // narrows that range and Bytes keeps the immutable allocation alive.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().add(offset).cast::<f32>(), len) }
}

impl SharedF32 {
    pub fn shared(bytes: bytes::Bytes, offset: usize, len: usize) -> Result<Self, &'static str> {
        let byte_len = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("float size overflow")?;
        let end = offset.checked_add(byte_len).ok_or("float range overflow")?;
        let raw = bytes.get(offset..end).ok_or("invalid float byte range")?;
        if !(raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
            return Err("misaligned float buffer");
        }
        Ok(Self(Storage::Shared { bytes, offset, len }))
    }

    pub fn is_shared(&self) -> bool {
        matches!(self.0, Storage::Shared { .. })
    }

    pub fn truncate(&mut self, len: usize) {
        if len >= self.len() {
            return;
        }
        match &mut self.0 {
            Storage::Owned(values) => values.truncate(len),
            Storage::Shared { len: current, .. } => *current = len,
        }
    }

    pub fn make_mut(&mut self) -> &mut Vec<f32> {
        self.0.owned_mut()
    }

    pub fn as_slice(&self) -> &[f32] {
        match &self.0 {
            Storage::Owned(values) => values.as_slice(),
            Storage::Shared { bytes, offset, len } => shared_slice(bytes, *offset, *len),
        }
    }
}

impl Deref for SharedF32 {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[f32]> for SharedF32 {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl From<Vec<f32>> for SharedF32 {
    fn from(values: Vec<f32>) -> Self {
        Self(Storage::Owned(values))
    }
}

impl fmt::Debug for SharedF32 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.as_slice()).finish()
    }
}

pub enum Embedding {
    Pooled(Vec<f32>),
    Shared(SharedF32),
    All(Vec<Vec<f32>>),
    SharedAll(Vec<SharedF32>),
}

pub type Embeddings = result_memory::ResultMap<Embedding>;
pub type Predictions = result_memory::ResultMap<Vec<f32>>;
pub type TokenPredictions = result_memory::ResultMap<Vec<Vec<f32>>>;

#[derive(Debug, Clone)]
pub enum DecisionInput {
    Laya {
        question_type: usize,
        markers: Vec<usize>,
    },
    OptionTokens {
        token_ids: Vec<u32>,
    },
    /// Joint Perplexity readouts: sequence-relative marker positions and valid option counts.
    InlineOptions {
        markers: Vec<usize>,
        option_counts: Vec<usize>,
    },
    /// OneJev block attention: prefix tokens followed by one question block.
    BlockOptionTokens {
        token_ids: Vec<u32>,
        prefix_length: usize,
    },
    Clef {
        fields: Vec<ClefField>,
    },
    /// Model-owned metadata for startup warmup and health probes.
    Warmup,
}

#[derive(Debug, Clone)]
pub struct ClefField {
    pub kind: usize,
    pub question: (usize, usize),
    pub options: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub struct DecisionOutput {
    pub logits: Vec<f32>,
    pub action_probability: f32,
    pub memory: Option<result_memory::ResultLease>,
}

impl DecisionOutput {
    /// Reserve a duplicate payload before copying an admitted result.
    pub fn try_clone(&self) -> Result<Self, BackendError> {
        let bytes = self
            .logits
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| BackendError::Inference("Decision clone size overflow".into()))?;
        let memory = self
            .memory
            .as_ref()
            .map(|memory| memory.reserve_more(bytes))
            .transpose()?;
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(self.logits.len())
            .map_err(|e| BackendError::Inference(e.to_string()))?;
        logits.extend_from_slice(&self.logits);
        Ok(Self {
            logits,
            action_probability: self.action_probability,
            memory,
        })
    }
}

/// An owned result, or CPU completion work after an ordered native submission.
/// Deferred work must retain all native buffers until `complete` or drop, and
/// bound pending work before returning. The next model call may run immediately.
pub enum Inference<T> {
    Ready(T),
    Deferred(Box<dyn FnOnce() -> Result<T, BackendError> + Send>),
}
impl<T: Send + 'static> Inference<T> {
    pub fn complete(self) -> Result<T, BackendError> {
        match self {
            Self::Ready(value) => Ok(value),
            Self::Deferred(complete) => complete(),
        }
    }
    pub fn try_map<U: Send + 'static>(
        self,
        map: impl FnOnce(T) -> Result<U, BackendError> + Send + 'static,
    ) -> Result<Inference<U>, BackendError> {
        match self {
            Self::Ready(value) => map(value).map(Inference::Ready),
            Self::Deferred(complete) => Ok(Inference::Deferred(Box::new(move || map(complete()?)))),
        }
    }
    pub fn map<U: Send + 'static>(self, map: impl FnOnce(T) -> U + Send + 'static) -> Inference<U> {
        match self {
            Self::Ready(value) => Inference::Ready(map(value)),
            Self::Deferred(complete) => Inference::Deferred(Box::new(move || complete().map(map))),
        }
    }
}

pub trait Backend {
    /// Execute internal startup/health input without populating user caches.
    fn probe(&self, batch: Batch, model_type: &ModelType) -> Result<(), BackendError> {
        match model_type {
            ModelType::Decision => {
                let count = batch.len();
                self.decide(batch, vec![DecisionInput::Warmup; count])
                    .map(|_| ())
            }
            ModelType::Classifier => self.predict(batch).map(|_| ()),
            ModelType::Embedding(_) => self.embed(batch).map(|_| ()),
        }
    }
    fn decide(
        &self,
        _batch: Batch,
        _inputs: Vec<DecisionInput>,
    ) -> Result<Vec<DecisionOutput>, BackendError> {
        Err(BackendError::Inference(
            "Model does not support typed decisions".into(),
        ))
    }
    fn health(&self) -> Result<(), BackendError>;
    /// Stop runtime-owned workers before the model is dropped.
    fn shutdown(&self) {}
    fn max_batch_size(&self) -> Option<usize> {
        None
    }

    fn supports_radix_mlp(&self) -> bool {
        false
    }

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError>;

    fn predict(&self, batch: Batch) -> Result<Predictions, BackendError>;

    fn predict_tokens(&self, batch: Batch) -> Result<TokenPredictions, BackendError>;

    fn submit_embed(&self, batch: Batch) -> Result<Inference<Embeddings>, BackendError> {
        self.embed(batch).map(Inference::Ready)
    }
    fn submit_predict(&self, batch: Batch) -> Result<Inference<Predictions>, BackendError> {
        self.predict(batch).map(Inference::Ready)
    }
    fn submit_predict_tokens(
        &self,
        batch: Batch,
    ) -> Result<Inference<TokenPredictions>, BackendError> {
        self.predict_tokens(batch).map(Inference::Ready)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ModelType {
    Decision,
    Classifier,
    Embedding(Pool),
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "snake_case")]
pub enum Pool {
    /// Select the CLS token as embedding
    Cls,
    /// Apply Mean pooling to the model embeddings
    Mean,
    /// Apply SPLADE (Sparse Lexical and Expansion) to the model embeddings.
    /// This option is only available if the loaded model is a `ForMaskedLM` Transformer
    /// model.
    Splade,
    /// Select the last token as embedding
    LastToken,
}

impl fmt::Display for Pool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Pool::Cls => write!(f, "cls"),
            Pool::Mean => write!(f, "mean"),
            Pool::Splade => write!(f, "splade"),
            Pool::LastToken => write!(f, "last_token"),
        }
    }
}

#[derive(Debug, Error, Clone)]
pub enum BackendError {
    #[error("{0}")]
    Overloaded(String),
    #[error("No backend found")]
    NoBackend,
    #[error("Could not start backend: {0}")]
    Start(String),
    #[error("{0}")]
    Inference(String),
    #[error("Backend is unhealthy")]
    Unhealthy,
    #[error("Weights not found: {0}")]
    WeightsNotFound(String),
}

#[cfg(test)]
mod tests {
    use super::SharedF32;
    use bytes::Bytes;

    fn aligned_buffer(values: &[f32]) -> Bytes {
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let align = std::mem::align_of::<f32>();
        let mut buffer = vec![0u8; payload.len() + align];
        let shift = (align - buffer.as_ptr() as usize % align) % align;
        buffer[shift..shift + payload.len()].copy_from_slice(&payload);
        Bytes::from(buffer).slice(shift..shift + payload.len())
    }

    #[test]
    fn rejects_misaligned_buffer() {
        let bytes = Bytes::from(vec![0u8; 16]);
        let align = std::mem::align_of::<f32>();
        let offset = (align - bytes.as_ptr() as usize % align) % align + 1;
        let sliced = bytes.slice(offset..offset + 4);
        // A relative offset of zero used to pass even though Bytes itself had
        // already been sliced to a misaligned address.
        assert!(SharedF32::shared(sliced.clone(), 0, 1).is_err());
        assert!(SharedF32::shared(sliced, 0, 0).is_err());
    }

    #[test]
    fn accepts_aligned_sliced_buffer() {
        let bytes = aligned_buffer(&[1.5, -2.25, 3.0]);
        let shared = SharedF32::shared(bytes, 0, 3).expect("aligned buffer is accepted");
        assert!(shared.is_shared());
        assert_eq!(shared.as_slice(), &[1.5, -2.25, 3.0]);
    }

    #[test]
    fn rejects_out_of_range_and_overflow() {
        let bytes = Bytes::from(vec![0u8; 16]);
        assert!(SharedF32::shared(bytes.clone(), 13, 2).is_err());
        assert!(SharedF32::shared(bytes.clone(), 0, usize::MAX).is_err());
        assert!(SharedF32::shared(bytes, usize::MAX, 1).is_err());
    }

    #[test]
    fn truncate_and_make_mut_isolate_shared_rows() {
        let bytes = aligned_buffer(&[1.0, 2.0, 3.0, 4.0]);
        let mut left = SharedF32::shared(bytes.clone(), 0, 4).expect("aligned buffer is accepted");
        let mut right = SharedF32::shared(bytes, 0, 4).expect("aligned buffer is accepted");

        left.truncate(2);
        assert_eq!(left.as_slice(), &[1.0, 2.0]);
        assert_eq!(right.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        right.make_mut()[0] = 9.0;
        assert!(!right.is_shared());
        assert_eq!(left.as_slice(), &[1.0, 2.0]);
        assert_eq!(right.as_slice(), &[9.0, 2.0, 3.0, 4.0]);
    }
}
