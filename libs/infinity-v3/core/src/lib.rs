//! Request preparation, batching and result delivery.
//!
//! Start at `infer::Infer` for the request path. `tokenization` prepares inputs,
//! `queue_actor` selects and packs batches, and `backend_pool` leases model
//! replicas. The backend crates run the models; core keeps request metadata
//! until results can be sent back to each caller.

pub mod backend_pool;
pub mod chat;
pub mod download;
mod fast_tokenization;
pub mod infer;
pub mod input;
pub mod input_memory;
pub mod multimodal;
pub mod queue_actor;
mod radix;
pub mod tokenization;

use infinity_v3_backend::BackendError;
use thiserror::Error;
use tokio::sync::TryAcquireError;

#[derive(Error, Debug)]
pub enum TextEmbeddingsError {
    #[error("tokenizer error {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("Input validation error: {0}")]
    Validation(String),
    #[error("Input validation error: {0}")]
    Empty(String),
    #[error("Model is overloaded")]
    Overloaded(#[from] TryAcquireError),
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),
}

pub(crate) fn log_response_dropped_after(stage: &'static str) {
    metrics::counter!("infinity_request_canceled", "stage" => stage).increment(1);
}
