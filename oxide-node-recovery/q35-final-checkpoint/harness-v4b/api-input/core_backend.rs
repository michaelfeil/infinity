pub mod block_attention_config;
mod multimodal;
pub mod result_memory;
#[cfg(feature = "clap")]
use clap::ValueEnum;
pub use multimodal::{AudioFeatures, ImagePatches, MultimodalEncoding};
use serde::Deserialize;
use std::fmt;
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

pub enum Embedding {
    Pooled(Vec<f32>),
    All(Vec<Vec<f32>>),
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
