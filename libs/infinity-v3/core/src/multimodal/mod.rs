//! Model-owned multimodal preparation before token budgeting and inference batching.
//!
//! The processor is a library entry point; HTTP/queue/backend wiring is a separate step.
mod image;
mod media;
mod positions;
mod qwen3_vl;

pub use qwen3_vl::{MultimodalConfig, PreparedMultimodal, PreparedQwenImages, Qwen3VlProcessor};
mod gemma4;
pub use gemma4::{Gemma4ImageProcessor, PreparedGemmaImages};

pub use positions::image_positions;

mod embedding_gemma2;
pub use embedding_gemma2::EmbeddingGemma2Processor;

mod d3;
pub use d3::{D3MediaProcessor, PreparedD3Media};
