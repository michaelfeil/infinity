//! Shared CPU-checkable media tensor, layout, position, and cold-load foundation.
//!
//! This module deliberately stops before backend kernels. It owns structural
//! admission for packed image patches, merged feature rows, audio features,
//! multimodal token spans, and dtype/device/strides metadata. It also provides
//! zero-copy borrow views and a pure cold-load chunk plan. Staging leases and
//! completion events stay in the backend that owns them; this module never
//! exposes a synchronous scratch-reuse shortcut that could race an async copy.
//!
//! Family-specific layouts stay explicit in [`policy`]. In particular, Qwen's
//! merge-group-major patch order is not shared with Mistral/Gemma/UltraVBERT.
//! No function here creates a CUDA context, enqueues work, falls back to CPU, or
//! imports Torch.

mod layout;
mod packed;
mod policy;
mod upload;

pub use layout::{F32View, MediaDevice, MediaDtype, MediaLayout};
pub use packed::{
    image_position_ids, AudioDescriptor, AudioView, FeatureRowSelection, ImageDescriptor,
    ImageView, PackedMedia, PackedMediaView, PackedPositions, PackedSequences, SequenceSpan,
};
pub use policy::{
    AudioLayoutPolicy, AudioSelection, ExtentPolicy, FramePolicy, ImageLayoutPolicy, MediaFamily,
    MergePolicy, PatchOrder,
};
pub use upload::{ColdChunk, ColdTensor, ColdUploadPlan};

use policy::MediaFamily as Family;

/// Typed rejection for malformed media, unsupported layouts, and checked
/// arithmetic failures. No variant implies that GPU work was submitted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MediaError {
    #[error("empty {0}")]
    Empty(&'static str),
    #[error("{0} has a zero dimension")]
    ZeroDimension(&'static str),
    #[error("{0} dimension product overflows usize")]
    DimensionOverflow(&'static str),
    #[error("{0} byte size overflows usize")]
    ByteOverflow(&'static str),
    #[error("{0}")]
    InvalidStride(&'static str),
    #[error("{0} arithmetic overflows")]
    LayoutOverflow(&'static str),
    #[error("media layout requires {required} elements but only {capacity} are available")]
    LayoutCapacity { required: usize, capacity: usize },
    #[error("invalid alignment {0}")]
    Alignment(usize),
    #[error("unsupported media or checkpoint dtype {name}")]
    UnsupportedDtype { name: String },
    #[error("host media views require CPU storage")]
    HostViewDevice,
    #[error("{family} does not accept {frames} frame(s): {allowed}")]
    UnsupportedFrames {
        family: Family,
        frames: usize,
        allowed: &'static str,
    },
    #[error("{family} requires {axis} {expected}, got {actual}")]
    UnsupportedGrid {
        family: Family,
        axis: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{family} merge size {merge} is invalid (maximum/fixed {max})")]
    InvalidMergeSize {
        family: Family,
        merge: usize,
        max: usize,
    },
    #[error("image grid {grid:?} is not divisible by merge size {merge}")]
    MergeMismatch { grid: [usize; 3], merge: usize },
    #[error("expected patch dimension {expected}, got {actual}")]
    PatchDimension { expected: usize, actual: usize },
    #[error("patch size {actual} is invalid (expected {expected}, fixed={fixed})")]
    PatchSize {
        expected: usize,
        actual: usize,
        fixed: bool,
    },
    #[error("expected payload length {expected}, got {actual}")]
    PayloadLength { expected: usize, actual: usize },
    #[error("token span start={token_start} count={token_count} is invalid for {tokens} tokens")]
    TokenSpan {
        token_start: usize,
        token_count: usize,
        tokens: usize,
    },
    #[error(
        "sequence start={sequence_start} length={sequence_length} is invalid for {tokens} tokens"
    )]
    SequenceSpan {
        sequence_start: usize,
        sequence_length: usize,
        tokens: usize,
    },
    #[error(
        "token span start={token_start} count={token_count} crosses sequence start={sequence_start} length={sequence_length}"
    )]
    SpanCrossesSequence {
        token_start: usize,
        token_count: usize,
        sequence_start: usize,
        sequence_length: usize,
    },
    #[error("media spans overlap at {next_start} after ending at {previous_end}")]
    OverlappingSpans {
        previous_end: usize,
        next_start: usize,
    },
    #[error("selection length {actual} does not match {expected}")]
    SelectionLength { expected: usize, actual: usize },
    #[error("selection index {index} is outside {limit}")]
    SelectionIndex { index: usize, limit: usize },
    #[error("selection destination {destination} is duplicated")]
    DuplicateDestination { destination: usize },
    #[error("position vector length {actual} does not match {expected}")]
    PositionLength { expected: usize, actual: usize },
    #[error("multimodal position exceeds u32")]
    PositionOverflow,
    #[error("audio feature length {actual} does not match {expected}")]
    AudioFeatureLength { expected: usize, actual: usize },
    #[error("audio token count {actual} does not match selected frame count {expected}")]
    AudioTokenCount { expected: usize, actual: usize },
    #[error("audio feature strides {actual:?} do not match policy {expected:?}")]
    AudioStride {
        expected: [usize; 2],
        actual: [usize; 2],
    },
    #[error("audio feature size {actual} does not match {expected}")]
    AudioFeatureSize { expected: usize, actual: usize },
    #[error("unsupported audio config feature_size={feature_size} strides={subsample_strides:?}")]
    UnsupportedAudioConfig {
        feature_size: usize,
        subsample_strides: [usize; 2],
    },
    #[error("audio requires at least one valid subsampled frame")]
    AudioEmptySelection,
    #[error("{family} does not have an audio media contract")]
    UnsupportedAudioFamily { family: Family },
    #[error("audio payload requires an explicit AudioLayoutPolicy")]
    MissingAudioPolicy,
    #[error("{0}")]
    Overflow(&'static str),
    #[error("cold upload chunk size {requested} is not a positive element multiple")]
    ColdChunkSize { requested: usize },
    #[error("cold source range for {tensor} offset={offset} bytes={bytes} exceeds {len}")]
    ColdSourceRange {
        tensor: String,
        offset: usize,
        bytes: usize,
        len: usize,
    },
}
