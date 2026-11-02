// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Bounded cold preparation and submission for the native Qwen3.5 Gated
//! DeltaNet operator ABI in `device/native_gdn/qwen35_gdn_native.cu`.
//!
//! This is an operator foundation, not an executor or a model route. It binds
//! the eight raw exports from one cold-loaded cubin, uploads the immutable
//! convolution/A_log/dt_bias/norm weights once, owns actual-shape reusable
//! scratch and metadata buffers, validates the real packed sequence geometry,
//! and submits the exact source arithmetic for the fresh and cached native
//! recurrence variants.
//!
//! Native initial state is FP32 `[sources, vh, 128, 128]` indexed by the
//! per-sequence source map. Final state is FP32 `[batch, vh, 128, 128]` indexed
//! by the sequence position, never by the source. The initial state is never
//! mutated. Z is accepted only with its exact contiguous 128-wide row stride;
//! noncontiguous Z is rejected rather than silently treated as merged-stride.
//!
//! `new_default_dense` additionally owns the ABI2 FlashInfer provider and the
//! separate three-export packing module. Every fresh request and native-shaped
//! cached suffix use that shared pipeline after the common convolution and
//! preparation; other cached shapes keep the raw recurrence. Chunk-scan,
//! reference arithmetic, Hopper prefill, quantized output, prefix checkpoints,
//! MoE and FP8 remain outside this bounded module. No fallback between them
//! occurs.

use crate::gdn_aot_provider::{DeviceFacts, GdnAotLaunch, GdnAotPlan, GdnAotProvider};
use crate::qwen35_work::{allocate_zero_ordered, ColdPending};
use cuda_core::{
    Context, DeviceBuffer, DeviceBufferView, DeviceCopy, Function, Module, RawDeviceBuffer, Stream,
};
use half::bf16;
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

#[path = "gdn_frame.rs"]
mod frame;
#[allow(unused_imports)] // Named by the unadmitted shared model integration.
pub(crate) use frame::GdnOwnedFrame;
pub use frame::GdnScratchBytes;

#[path = "gdn_block_frame.rs"]
mod block_frame;
#[allow(unused_imports)] // Consumed by the unadmitted Qwen3.5 B1 block integration.
pub(crate) use block_frame::{
    GdnBlockFrame, GdnBlockMetadata, GdnBlockOutput, GdnBlockPair, GdnBlockSuffix,
};

/// Native key width of every GDN head.
pub const GDN_KEY_DIM: usize = 128;
/// Native value width of every GDN head.
pub const GDN_VALUE_DIM: usize = 128;
/// Largest admitted value-head count in the native ABI.
pub const GDN_MAX_VALUE_HEADS: usize = 128;
/// Largest admitted depthwise convolution kernel.
pub const GDN_MAX_KERNEL: usize = 16;
/// Largest int32 token extent accepted by the native kernels.
pub const GDN_MAX_TOKENS: usize = i32::MAX as usize;

const REQUIRED_EXPORTS: [&str; 8] = [
    "qwen35_conv_bf16",
    "qwen35_gdn_prepare",
    "qwen35_gdn_recurrent",
    "qwen35_gdn_norm",
    "infinity_qwen35_packed_conv_fresh",
    "infinity_qwen35_packed_conv_cached",
    "infinity_qwen35_gdn_cached_lanes8",
    "infinity_qwen35_gdn_cached_lanes16",
];
const REQUIRED_ARCHITECTURES: [&str; 3] = ["sm_90", "sm_100", "sm_120"];
const REQUIRED_FLAGS: [&str; 9] = [
    "-std=c++17",
    "-O3",
    "--cubin",
    "--fmad=true",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "-Xptxas",
    "-v",
];
const REQUIRED_SOURCES: [&str; 3] = ["qwen35_gdn_native.cu", "NOTICE", "LICENSE.candle-MIT"];
const GDN_NATIVE_SOURCE: &[u8] = include_bytes!("../device/native_gdn/qwen35_gdn_native.cu");
const AOT_REQUIRED_EXPORTS: [&str; 3] = [
    "infinity_qwen35_gdn_aot_pack_bf16",
    "infinity_qwen35_gdn_aot_state_kv_to_vk",
    "infinity_qwen35_gdn_aot_state_vk_to_kv",
];
const AOT_REQUIRED_SOURCES: [&str; 4] = [
    "qwen35_gdn_aot.cu",
    "NOTICE",
    "LICENSE",
    "LICENSE.candle-MIT",
];
const AOT_OPERATOR: &str = "qwen35_gdn_aot_pack";
const GDN_AOT_SOURCE: &[u8] = include_bytes!("../device/native_gdn_aot/qwen35_gdn_aot.cu");
const AOT_PACK_THREADS: u32 = 256;
const AOT_STATE_TILE: usize = 32;
const AOT_STATE_BLOCK: (u32, u32, u32) = (32, 32, 1);
const PREPARE_THREADS: u32 = 128;
const RECURRENT_THREADS: u32 = 256;
const CACHED_THREADS: u32 = 512;
const PACKED_CONV_THREADS: u32 = 256;
const NORM_THREADS: u32 = 128;
const MAX_GRID_X: usize = i32::MAX as usize;
const MAX_GRID_Y: usize = u16::MAX as usize;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide Qwen3.5 GDN prepared: {error}"))
}

fn infer(error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide Qwen3.5 GDN prepared: {error}"))
}

fn grid_u32(value: usize, name: &str) -> Result<u32, BackendError> {
    u32::try_from(value).map_err(|_| infer(format!("{name} {value} exceeds the native u32 grid")))
}

fn int32_value(value: usize, name: &str) -> Result<i32, BackendError> {
    i32::try_from(value).map_err(|_| infer(format!("{name} {value} exceeds the native int32 ABI")))
}

fn product(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| infer(format!("{name} product overflows usize")))
}

/// Validate the one norm-epsilon value accepted by the bounded device ABI.
pub fn validate_epsilon(epsilon: f32) -> Result<(), BackendError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(start(format!(
            "norm epsilon must be finite and positive, found {epsilon}"
        )));
    }
    Ok(())
}

/// The `[Q|K|V]` head geometry and depthwise convolution width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnGeometry {
    key_heads: usize,
    value_heads: usize,
    kernel: usize,
}

impl GdnGeometry {
    /// Pure CPU admission of the native head geometry.
    pub fn new(key_heads: usize, value_heads: usize, kernel: usize) -> Result<Self, BackendError> {
        if key_heads == 0 {
            return Err(start("key head count must be positive"));
        }
        if value_heads == 0 {
            return Err(start("value head count must be positive"));
        }
        if value_heads < key_heads {
            return Err(start(format!(
                "value heads {value_heads} must be at least key heads {key_heads}"
            )));
        }
        if value_heads > GDN_MAX_VALUE_HEADS {
            return Err(start(format!(
                "value heads {value_heads} exceed the native maximum {GDN_MAX_VALUE_HEADS}"
            )));
        }
        if !value_heads.is_multiple_of(key_heads) {
            return Err(start(format!(
                "value heads {value_heads} are not divisible by key heads {key_heads}"
            )));
        }
        if kernel == 0 || kernel > GDN_MAX_KERNEL {
            return Err(start(format!(
                "convolution kernel {kernel} is outside 1..={GDN_MAX_KERNEL}"
            )));
        }
        Ok(Self {
            key_heads,
            value_heads,
            kernel,
        })
    }

    pub const fn key_heads(self) -> usize {
        self.key_heads
    }

    pub const fn value_heads(self) -> usize {
        self.value_heads
    }

    pub const fn kernel(self) -> usize {
        self.kernel
    }

    /// Checked packed `[Q|K|V]` channel width.
    pub fn channels(self) -> usize {
        (2 * self.key_heads + self.value_heads) * GDN_KEY_DIM
    }

    pub const fn qk_width(self) -> usize {
        2 * GDN_KEY_DIM
    }

    pub const fn state_row_width(self) -> usize {
        GDN_KEY_DIM * GDN_VALUE_DIM
    }

    /// Source recurrence policy: eight lanes at batch*value_heads >= 256.
    pub fn cached_lanes(self, sequences: usize) -> Result<u32, BackendError> {
        let work = product(sequences, self.value_heads, "cached recurrence head work")?;
        if work >= 256 {
            Ok(8)
        } else {
            Ok(16)
        }
    }

    fn packed_conv_grid(self, tokens: usize) -> Result<(u32, u32, u32), BackendError> {
        if tokens > MAX_GRID_X {
            return Err(start(format!(
                "packed convolution token grid {tokens} exceeds {MAX_GRID_X}"
            )));
        }
        let blocks = (self.channels() / 4).div_ceil(PACKED_CONV_THREADS as usize);
        if blocks > MAX_GRID_Y {
            return Err(start(format!(
                "packed convolution column grid {blocks} exceeds {MAX_GRID_Y}"
            )));
        }
        Ok((
            grid_u32(tokens, "packed convolution tokens")?,
            grid_u32(blocks, "packed convolution columns")?,
            1,
        ))
    }
}

/// Maximum reusable extents owned by one prepared operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnCapacity {
    tokens: usize,
    sequences: usize,
    history_sequences: usize,
}

impl GdnCapacity {
    pub fn new(
        tokens: usize,
        sequences: usize,
        history_sequences: usize,
    ) -> Result<Self, BackendError> {
        if tokens == 0 {
            return Err(start("prepared token capacity must be positive"));
        }
        if sequences == 0 {
            return Err(start("prepared sequence capacity must be positive"));
        }
        if tokens > GDN_MAX_TOKENS {
            return Err(start(format!(
                "prepared token capacity {tokens} exceeds {GDN_MAX_TOKENS}"
            )));
        }
        if sequences > tokens {
            return Err(start(format!(
                "prepared sequence capacity {sequences} exceeds token capacity {tokens}"
            )));
        }
        if history_sequences > GDN_MAX_TOKENS {
            return Err(start(format!(
                "prepared history sequence capacity {history_sequences} exceeds {GDN_MAX_TOKENS}"
            )));
        }
        Ok(Self {
            tokens,
            sequences,
            history_sequences,
        })
    }

    pub const fn tokens(self) -> usize {
        self.tokens
    }

    pub const fn sequences(self) -> usize {
        self.sequences
    }

    pub const fn history_sequences(self) -> usize {
        self.history_sequences
    }
}

/// Every reusable scratch/metadata extent for one capacity, checked at load.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnScratchLayout {
    convolution: usize,
    qk: usize,
    decay_beta: usize,
    recurrence: usize,
    output: usize,
    final_state: usize,
    cumulative: usize,
    sources: usize,
    history_cumulative: usize,
    channels: usize,
}

impl GdnScratchLayout {
    pub fn new(geometry: GdnGeometry, capacity: GdnCapacity) -> Result<Self, BackendError> {
        let channels = geometry.channels();
        let convolution = product(capacity.tokens, channels, "convolution scratch")?;
        if convolution > u32::MAX as usize {
            return Err(start(format!(
                "native token*channel extent {convolution} exceeds UINT32_MAX"
            )));
        }
        let qk_rows = product(capacity.tokens, geometry.key_heads, "qk row")?;
        if qk_rows > MAX_GRID_X {
            return Err(start(format!(
                "prepare grid {qk_rows} exceeds {MAX_GRID_X}"
            )));
        }
        let qk = product(qk_rows, geometry.qk_width(), "qk scratch")?;
        let decay_beta = product(
            product(capacity.tokens, geometry.value_heads, "gate head")?,
            2,
            "gate scratch",
        )?;
        let recurrence = product(
            product(capacity.tokens, geometry.value_heads, "recurrence head")?,
            GDN_VALUE_DIM,
            "recurrence scratch",
        )?;
        let norm_rows = product(capacity.tokens, geometry.value_heads, "norm rows")?;
        if norm_rows > MAX_GRID_X {
            return Err(start(format!("norm grid {norm_rows} exceeds {MAX_GRID_X}")));
        }
        let final_state = product(
            product(capacity.sequences, geometry.value_heads, "final state head")?,
            geometry.state_row_width(),
            "final state scratch",
        )?;
        let cumulative = capacity
            .sequences
            .checked_add(1)
            .ok_or_else(|| infer("cumulative sequence metadata length overflows"))?;
        let history_cumulative = capacity
            .history_sequences
            .checked_add(1)
            .ok_or_else(|| infer("history cumulative metadata length overflows"))?;
        Ok(Self {
            convolution,
            qk,
            decay_beta,
            recurrence,
            output: recurrence,
            final_state,
            cumulative,
            sources: capacity.sequences,
            history_cumulative,
            channels,
        })
    }

    pub const fn convolution_elements(self) -> usize {
        self.convolution
    }

    pub const fn qk_elements(self) -> usize {
        self.qk
    }

    pub const fn decay_beta_elements(self) -> usize {
        self.decay_beta
    }

    pub const fn recurrence_elements(self) -> usize {
        self.recurrence
    }

    pub const fn output_elements(self) -> usize {
        self.output
    }

    pub const fn final_state_elements(self) -> usize {
        self.final_state
    }

    pub const fn cumulative_elements(self) -> usize {
        self.cumulative
    }

    pub const fn source_elements(self) -> usize {
        self.sources
    }

    pub const fn history_cumulative_elements(self) -> usize {
        self.history_cumulative
    }

    pub const fn channels(self) -> usize {
        self.channels
    }
}

/// One actual request's strictly increasing packed sequence boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnPlan {
    geometry: GdnGeometry,
    tokens: usize,
    sequences: usize,
    max_sequence: usize,
}

impl GdnPlan {
    /// Admit real `cu` metadata: exactly starts at zero, strictly increases,
    /// ends at the actual token count, and fits the prepared capacity.
    pub fn new(
        geometry: GdnGeometry,
        capacity: GdnCapacity,
        cumulative: &[u32],
    ) -> Result<Self, BackendError> {
        if cumulative.len() < 2 {
            return Err(start(
                "cumulative sequence metadata must contain at least one sequence",
            ));
        }
        let sequences = cumulative.len() - 1;
        if sequences > capacity.sequences {
            return Err(infer(format!(
                "request sequences {sequences} exceed prepared capacity {}",
                capacity.sequences
            )));
        }
        if cumulative.first().copied() != Some(0) {
            return Err(infer("cumulative sequence metadata must start at zero"));
        }
        let mut previous = 0u32;
        let mut max_sequence = 0usize;
        for (index, &current) in cumulative.iter().enumerate().skip(1) {
            if current <= previous {
                return Err(infer(format!(
                    "cumulative sequence boundaries are not strictly increasing at index {index}"
                )));
            }
            if current as usize > GDN_MAX_TOKENS {
                return Err(infer(format!(
                    "sequence boundary {current} exceeds the native int32 token bound"
                )));
            }
            max_sequence = max_sequence.max((current - previous) as usize);
            previous = current;
        }
        let tokens = previous as usize;
        if tokens > capacity.tokens {
            return Err(infer(format!(
                "request tokens {tokens} exceed prepared capacity {}",
                capacity.tokens
            )));
        }
        let _ = GdnScratchLayout::new(geometry, capacity)?;
        Ok(Self {
            geometry,
            tokens,
            sequences,
            max_sequence,
        })
    }

    pub const fn geometry(self) -> GdnGeometry {
        self.geometry
    }

    pub const fn tokens(self) -> usize {
        self.tokens
    }

    pub const fn sequences(self) -> usize {
        self.sequences
    }

    pub const fn max_sequence(self) -> usize {
        self.max_sequence
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GdnRequestExtents {
    output_elements: usize,
    final_elements: Option<usize>,
}

fn request_extents(
    geometry: GdnGeometry,
    plan: GdnPlan,
    write_final_state: bool,
) -> Result<GdnRequestExtents, BackendError> {
    let output_elements = product(
        product(plan.tokens, geometry.value_heads, "output rows")?,
        GDN_VALUE_DIM,
        "output extent",
    )?;
    let final_elements = if write_final_state {
        Some(product(
            plan.sequences,
            product(
                geometry.value_heads,
                geometry.state_row_width(),
                "final state row",
            )?,
            "request final state",
        )?)
    } else {
        None
    };
    Ok(GdnRequestExtents {
        output_elements,
        final_elements,
    })
}

/// Device-buffer extent facts used by the bounded input-layout admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnTensorExtent {
    pub elements: usize,
    pub row_stride: usize,
    pub byte_alignment: usize,
}

/// Admitted packed input strides. Z is deliberately restricted to contiguous
/// 128-wide rows; noncontiguous Z is outside this bounded ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnInputLayout {
    pub qkv_row_stride: usize,
    pub z_row_stride: usize,
    pub ab_row_stride: usize,
}

impl GdnInputLayout {
    pub fn admit(
        geometry: GdnGeometry,
        tokens: usize,
        qkv: GdnTensorExtent,
        z: GdnTensorExtent,
        ab: GdnTensorExtent,
    ) -> Result<Self, BackendError> {
        if tokens == 0 {
            return Err(infer("input layout requires at least one token"));
        }
        let channels = geometry.channels();
        if qkv.row_stride < channels {
            return Err(infer(format!(
                "qkv row stride {} is smaller than the {channels}-channel row",
                qkv.row_stride
            )));
        }
        if !qkv.row_stride.is_multiple_of(4) {
            return Err(infer(format!(
                "qkv row stride {} is not a multiple of four BF16 elements",
                qkv.row_stride
            )));
        }
        if qkv.byte_alignment < 8 {
            return Err(infer(format!(
                "qkv base alignment {} is below the required eight bytes",
                qkv.byte_alignment
            )));
        }
        let qkv_needed = product(tokens.saturating_sub(1), qkv.row_stride, "qkv extent")?
            .checked_add(channels)
            .ok_or_else(|| infer("qkv extent overflows usize"))?;
        if qkv.elements < qkv_needed {
            return Err(infer(format!(
                "qkv extent {} is shorter than the {qkv_needed} elements addressed",
                qkv.elements
            )));
        }

        let z_width = product(geometry.value_heads, GDN_VALUE_DIM, "z width")?;
        if z.row_stride != z_width {
            return Err(infer(format!(
                "z row stride must be exactly {z_width}; noncontiguous Z is rejected"
            )));
        }
        if z.byte_alignment < 2 {
            return Err(infer("z base alignment is below two bytes"));
        }
        let z_needed = product(tokens, z_width, "z extent")?;
        if z.elements < z_needed {
            return Err(infer(format!(
                "z extent {} is shorter than the {z_needed} elements addressed",
                z.elements
            )));
        }

        let ab_width = product(geometry.value_heads, 2, "a/b width")?;
        if ab.row_stride != ab_width {
            return Err(infer(format!(
                "a/b row stride {} must be exactly {ab_width}",
                ab.row_stride
            )));
        }
        if ab.byte_alignment < 2 {
            return Err(infer("a/b base alignment is below two bytes"));
        }
        let ab_needed = product(tokens, ab_width, "a/b extent")?;
        if ab.elements < ab_needed {
            return Err(infer(format!(
                "a/b extent {} is shorter than the {ab_needed} elements addressed",
                ab.elements
            )));
        }

        Ok(Self {
            qkv_row_stride: qkv.row_stride,
            z_row_stride: z.row_stride,
            ab_row_stride: ab.row_stride,
        })
    }
}

/// Validated resident initial-state/history mapping for one cached request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnHistoryPlan {
    initial_sequences: usize,
    history_rows: usize,
    initial_elements: usize,
    history_elements: usize,
    final_elements: usize,
}

impl GdnHistoryPlan {
    pub fn new(
        geometry: GdnGeometry,
        sequences: usize,
        sources: &[i64],
        history_cumulative: &[i32],
        history_rows: usize,
    ) -> Result<Self, BackendError> {
        if sequences == 0 {
            return Err(infer("cached history requires at least one sequence"));
        }
        if sources.len() != sequences {
            return Err(infer(format!(
                "cached source map has {} entries for {sequences} sequences",
                sources.len()
            )));
        }
        if history_cumulative.len() < 2 {
            return Err(infer(
                "cached history cumulative metadata must contain at least one source",
            ));
        }
        if history_cumulative.first().copied() != Some(0) {
            return Err(infer(
                "cached history cumulative metadata must start at zero",
            ));
        }
        let initial_sequences = history_cumulative.len() - 1;
        let mut previous = 0i64;
        for (index, &current) in history_cumulative.iter().enumerate().skip(1) {
            if current < 0 || i64::from(current) < previous {
                return Err(infer(format!(
                    "cached history boundaries decrease at index {index}"
                )));
            }
            let length = i64::from(current) - previous;
            if length as usize > geometry.kernel.saturating_sub(1) {
                return Err(infer(format!(
                    "cached history source {index} carries {length} rows, more than kernel-1"
                )));
            }
            previous = i64::from(current);
        }
        let expected_rows = usize::try_from(previous)
            .map_err(|_| infer("cached history row count exceeds usize"))?;
        if expected_rows != history_rows {
            return Err(infer(format!(
                "cached history has {history_rows} rows, cumulative metadata ends at {expected_rows}"
            )));
        }
        for (sequence, &source) in sources.iter().enumerate() {
            if source < 0 || source as usize >= initial_sequences {
                return Err(infer(format!(
                    "cached sequence {sequence} source {source} is outside 0..{initial_sequences}"
                )));
            }
        }

        let state_elements = product(
            geometry.value_heads,
            geometry.state_row_width(),
            "state row extent",
        )?;
        let initial_elements = product(initial_sequences, state_elements, "initial state extent")?;
        let history_elements = product(history_rows, geometry.channels(), "history extent")?;
        let final_elements = product(sequences, state_elements, "final state extent")?;
        Ok(Self {
            initial_sequences,
            history_rows,
            initial_elements,
            history_elements,
            final_elements,
        })
    }

    pub const fn initial_sequences(self) -> usize {
        self.initial_sequences
    }

    pub const fn history_rows(self) -> usize {
        self.history_rows
    }

    pub const fn initial_elements(self) -> usize {
        self.initial_elements
    }

    pub const fn history_elements(self) -> usize {
        self.history_elements
    }

    pub const fn final_elements(self) -> usize {
        self.final_elements
    }
}

fn history_row_count(elements: usize, channels: usize) -> Result<usize, BackendError> {
    if channels == 0 || !elements.is_multiple_of(channels) {
        return Err(infer(format!(
            "cached history extent {elements} is not a whole number of {channels}-channel rows"
        )));
    }
    Ok(elements / channels)
}

/// Element offset of a sequence-indexed final state row. The source map never
/// participates in final-state addressing.
pub fn final_state_offset(geometry: GdnGeometry, sequence: usize) -> Result<usize, BackendError> {
    let row = product(
        geometry.value_heads,
        geometry.state_row_width(),
        "final state row",
    )?;
    product(sequence, row, "final state offset")
}

/// Exact packed-buffer extents owned once by the default dense AOT path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GdnAotLayout {
    packed_qk: usize,
    packed_v: usize,
    gates: usize,
    state: usize,
}

impl GdnAotLayout {
    fn new(geometry: GdnGeometry, capacity: GdnCapacity) -> Result<Self, BackendError> {
        let packed_qk = product(
            product(capacity.tokens, geometry.key_heads, "AOT packed Q/K rows")?,
            GDN_KEY_DIM,
            "AOT packed Q/K",
        )?;
        let packed_v = product(
            product(capacity.tokens, geometry.value_heads, "AOT packed V rows")?,
            GDN_KEY_DIM,
            "AOT packed V",
        )?;
        let gates = product(capacity.tokens, geometry.value_heads, "AOT gates")?;
        let state = product(
            product(capacity.sequences, geometry.value_heads, "AOT state heads")?,
            geometry.state_row_width(),
            "AOT state",
        )?;
        let _ = aot_state_tile_count(
            capacity.sequences,
            geometry.value_heads,
            GDN_KEY_DIM,
            GDN_VALUE_DIM,
        )?;
        let _ = product(state, std::mem::size_of::<f32>(), "AOT state bytes")?;
        Ok(Self {
            packed_qk,
            packed_v,
            gates,
            state,
        })
    }
}

/// Native `state_tile_count` grid extent with the provider's u32 launch bound.
fn aot_state_tile_count(
    batch: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<u32, BackendError> {
    if batch == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 {
        return Err(infer("AOT state transpose requires positive extents"));
    }
    let tiles_key = key_dim.div_ceil(AOT_STATE_TILE);
    let tiles_value = value_dim.div_ceil(AOT_STATE_TILE);
    let tiles = product(batch, value_heads, "AOT state batch*heads")?;
    let tiles = product(tiles, tiles_value, "AOT state value tiles")?;
    let tiles = product(tiles, tiles_key, "AOT state key tiles")?;
    if tiles > MAX_GRID_X {
        return Err(infer("AOT state tile grid exceeds the CUDA gridDimX limit"));
    }
    u32::try_from(tiles).map_err(|_| infer(format!("AOT state tile grid {tiles} exceeds u32")))
}

/// Native pack grid: one token block and 256-column value-wide blocks.
fn aot_pack_grid(tokens: usize, value_heads: usize) -> Result<(u32, u32, u32), BackendError> {
    let columns = product(value_heads, GDN_KEY_DIM, "AOT pack columns")?;
    let blocks = columns.div_ceil(AOT_PACK_THREADS as usize);
    Ok((
        grid_u32(tokens, "AOT pack tokens")?,
        grid_u32(blocks, "AOT pack column blocks")?,
        1,
    ))
}

/// Native selection predicate after the shared convolution and prepare work.
/// Fresh state always admits the provider; a cached suffix only does so for the
/// native chunk-parallel shape.
fn aot_admits(cached: bool, tokens: usize, sequences: usize) -> bool {
    !cached || (tokens >= 256 && tokens / sequences >= 64)
}

/// A checked borrowed range with no allocation or reference-count operation.
/// It does not prolong device storage lifetime; the engine-owned Work must do so.
#[derive(Clone, Copy)]
pub struct GdnBufferView<'a, T> {
    storage: &'a RawDeviceBuffer,
    pointer: u64,
    elements: usize,
    _typed: std::marker::PhantomData<&'a T>,
}

impl<'a, T> GdnBufferView<'a, T> {
    pub fn from_buffer(buffer: &'a DeviceBuffer<T>) -> Self {
        Self {
            storage: buffer.storage(),
            pointer: buffer.cu_deviceptr(),
            elements: buffer.len(),
            _typed: std::marker::PhantomData,
        }
    }

    pub fn from_buffer_range(
        buffer: &'a DeviceBuffer<T>,
        range: std::ops::Range<usize>,
    ) -> Result<Self, BackendError> {
        let (pointer, elements) = borrowed_range(
            buffer.cu_deviceptr(),
            buffer.len(),
            range,
            std::mem::size_of::<T>(),
        )?;
        Ok(Self {
            storage: buffer.storage(),
            pointer,
            elements,
            _typed: std::marker::PhantomData,
        })
    }

    pub fn from_view(view: &'a DeviceBufferView<'_, T>) -> Self {
        Self {
            storage: view.storage(),
            pointer: view.cu_deviceptr(),
            elements: view.len(),
            _typed: std::marker::PhantomData,
        }
    }

    pub const fn cu_deviceptr(&self) -> u64 {
        self.pointer
    }
    pub const fn len(&self) -> usize {
        self.elements
    }
    pub const fn is_empty(&self) -> bool {
        self.elements == 0
    }
    fn context(&self) -> &Arc<Context> {
        self.storage.context()
    }
    fn storage(&self) -> &RawDeviceBuffer {
        self.storage
    }
}

// Both types are transparent GPU BF16 bit containers; these compile-time
// layout checks guard the cross-module borrowed binding without a host cast.
const _: [(); 2] = [(); std::mem::size_of::<bf16>()];
const _: [(); 2] = [(); std::mem::size_of::<crate::tensor::Bf16>()];
const _: [(); 2] = [(); std::mem::align_of::<bf16>()];
const _: [(); 2] = [(); std::mem::align_of::<crate::tensor::Bf16>()];

impl<'a> GdnBufferView<'a, bf16> {
    /// Borrow the framework's other transparent BF16 representation. The
    /// allocation is unchanged; only its checked device byte range is bound.
    pub fn from_tensor_buffer_range(
        buffer: &'a DeviceBuffer<crate::tensor::Bf16>,
        range: std::ops::Range<usize>,
    ) -> Result<Self, BackendError> {
        let (pointer, elements) = borrowed_range(buffer.cu_deviceptr(), buffer.len(), range, 2)?;
        Ok(Self {
            storage: buffer.storage(),
            pointer,
            elements,
            _typed: std::marker::PhantomData,
        })
    }

    pub fn from_tensor_buffer(buffer: &'a DeviceBuffer<crate::tensor::Bf16>) -> Self {
        Self {
            storage: buffer.storage(),
            pointer: buffer.cu_deviceptr(),
            elements: buffer.len(),
            _typed: std::marker::PhantomData,
        }
    }
}

fn borrowed_range(
    pointer: u64,
    elements: usize,
    range: std::ops::Range<usize>,
    element_size: usize,
) -> Result<(u64, usize), BackendError> {
    if element_size == 0 || range.start > range.end || range.end > elements {
        return Err(infer("borrowed GDN range exceeds its typed allocation"));
    }
    let bytes = range
        .start
        .checked_mul(element_size)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| infer("borrowed GDN range byte offset overflows"))?;
    let pointer = pointer
        .checked_add(bytes)
        .ok_or_else(|| infer("borrowed GDN device pointer overflows"))?;
    Ok((pointer, range.end - range.start))
}

/// Engine inputs may borrow subranges of phase-reused arenas. Logical Z and
/// A/B rows retain the same contiguous contracts as standalone DeviceBuffers.
#[derive(Clone, Copy)]
pub struct GdnInputViews<'a> {
    pub qkv: GdnBufferView<'a, bf16>,
    pub qkv_row_stride: usize,
    pub z: GdnBufferView<'a, bf16>,
    pub z_row_stride: usize,
    pub ab: GdnBufferView<'a, bf16>,
    pub ab_row_stride: usize,
}

/// Immutable packed input views for one submission.
pub struct GdnInputs<'a> {
    pub qkv: &'a DeviceBuffer<bf16>,
    pub qkv_row_stride: usize,
    pub z: &'a DeviceBuffer<bf16>,
    pub z_row_stride: usize,
    pub ab: &'a DeviceBuffer<bf16>,
    pub ab_row_stride: usize,
}

impl<'a> GdnInputs<'a> {
    pub fn borrowed(&self) -> GdnInputViews<'a> {
        GdnInputViews {
            qkv: GdnBufferView::from_buffer(self.qkv),
            qkv_row_stride: self.qkv_row_stride,
            z: GdnBufferView::from_buffer(self.z),
            z_row_stride: self.z_row_stride,
            ab: GdnBufferView::from_buffer(self.ab),
            ab_row_stride: self.ab_row_stride,
        }
    }
}

impl GdnInputViews<'_> {
    fn qkv_extent(&self) -> GdnTensorExtent {
        tensor_extent(self.qkv.cu_deviceptr(), self.qkv.len(), self.qkv_row_stride)
    }

    fn z_extent(&self) -> GdnTensorExtent {
        tensor_extent(self.z.cu_deviceptr(), self.z.len(), self.z_row_stride)
    }

    fn ab_extent(&self) -> GdnTensorExtent {
        tensor_extent(self.ab.cu_deviceptr(), self.ab.len(), self.ab_row_stride)
    }
}

fn tensor_extent(pointer: u64, elements: usize, row_stride: usize) -> GdnTensorExtent {
    GdnTensorExtent {
        elements,
        row_stride,
        byte_alignment: pointer_alignment(pointer),
    }
}

fn pointer_alignment(pointer: u64) -> usize {
    if pointer == 0 {
        0
    } else {
        usize::try_from(pointer & pointer.wrapping_neg()).unwrap_or(usize::MAX)
    }
}

/// One cached history + immutable initial state.
pub struct GdnHistory<'a> {
    pub rows: &'a DeviceBuffer<bf16>,
    pub cumulative: &'a [i32],
    pub sources: &'a [i64],
}

pub struct GdnCachedState<'a> {
    pub initial: &'a DeviceBuffer<f32>,
    pub history: GdnHistory<'a>,
}

/// Fresh (zero initial state) or cached (immutable initial + source map).
pub enum GdnState<'a> {
    Fresh,
    Cached(GdnCachedState<'a>),
}

/// Crate-private borrowed metadata for an already-owned state. It never takes
/// ownership of or prolongs the device allocations; the engine-owned Work that
/// produced the views must outlive every use.
#[derive(Clone, Copy)]
pub(crate) enum GdnStateViews<'a> {
    Fresh,
    Cached {
        initial: GdnBufferView<'a, f32>,
        history_rows: GdnBufferView<'a, bf16>,
        history_cumulative: &'a [i32],
        sources: &'a [i64],
    },
}

impl<'a> GdnState<'a> {
    pub(crate) fn borrowed<'b>(&'b self) -> GdnStateViews<'b> {
        match self {
            GdnState::Fresh => GdnStateViews::Fresh,
            GdnState::Cached(cached) => GdnStateViews::Cached {
                initial: GdnBufferView::from_buffer(cached.initial),
                history_rows: GdnBufferView::from_buffer(cached.history.rows),
                history_cumulative: cached.history.cumulative,
                sources: cached.history.sources,
            },
        }
    }
}

/// One-time upload slices. Lengths are checked against the exact geometry.
pub struct GdnWeights<'a> {
    pub conv: &'a [bf16],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [bf16],
}

/// The eight cold-loaded exports from one validated bundle manifest.
pub struct GdnKernelSet {
    context: Arc<Context>,
    #[allow(dead_code)]
    conv: Function,
    prepare: Function,
    recurrent: Function,
    norm: Function,
    packed_conv_fresh: Function,
    packed_conv_cached: Function,
    cached_lanes8: Function,
    cached_lanes16: Function,
}

impl GdnKernelSet {
    /// Validate every manifest record, source digest, and cubin on the CPU
    /// before any CUDA context is created or queried.
    pub fn validate_bundle(bundle: &Path) -> Result<(), BackendError> {
        GdnValidatedBundle::load(bundle).map(|_| ())
    }

    /// Read and validate `manifest.json`, select the exact SM90/100/120 cubin
    /// for this context, and bind every required export. This is cold work.
    pub fn load_bundle(context: &Arc<Context>, bundle: &Path) -> Result<Self, BackendError> {
        let validated = GdnValidatedBundle::load(bundle)?;
        let (major, minor) = context.compute_capability().map_err(start)?;
        let architecture = format!("sm_{major}{minor}");
        let image = validated.image(&architecture).ok_or_else(|| {
            start(format!(
                "bundle has no {architecture} cubin for this context"
            ))
        })?;
        let module = context.load_module_from_image(image).map_err(start)?;
        Self::from_module(&module)
    }

    /// Bind every required export from an already loaded module.
    fn from_module(module: &Arc<Module>) -> Result<Self, BackendError> {
        Ok(Self {
            context: module.context().clone(),
            conv: module.load_function("qwen35_conv_bf16").map_err(start)?,
            prepare: module.load_function("qwen35_gdn_prepare").map_err(start)?,
            recurrent: module
                .load_function("qwen35_gdn_recurrent")
                .map_err(start)?,
            norm: module.load_function("qwen35_gdn_norm").map_err(start)?,
            packed_conv_fresh: module
                .load_function("infinity_qwen35_packed_conv_fresh")
                .map_err(start)?,
            packed_conv_cached: module
                .load_function("infinity_qwen35_packed_conv_cached")
                .map_err(start)?,
            cached_lanes8: module
                .load_function("infinity_qwen35_gdn_cached_lanes8")
                .map_err(start)?,
            cached_lanes16: module
                .load_function("infinity_qwen35_gdn_cached_lanes16")
                .map_err(start)?,
        })
    }
}

/// The three additive AOT exports from the separate packing bundle.
struct GdnAotKernelSet {
    #[allow(dead_code)]
    context: Arc<Context>,
    pack: Function,
    kv_to_vk: Function,
    vk_to_kv: Function,
}

impl GdnAotKernelSet {
    fn bind(
        context: &Arc<Context>,
        validated: &GdnAotValidatedBundle,
    ) -> Result<Self, BackendError> {
        let (major, minor) = context.compute_capability().map_err(start)?;
        let architecture = format!("sm_{major}{minor}");
        let image = validated.image(&architecture).ok_or_else(|| {
            start(format!(
                "AOT packing bundle has no {architecture} cubin for this context"
            ))
        })?;
        let module = context.load_module_from_image(image).map_err(start)?;
        Self::from_module(&module)
    }

    fn from_module(module: &Arc<Module>) -> Result<Self, BackendError> {
        Ok(Self {
            context: module.context().clone(),
            pack: module
                .load_function("infinity_qwen35_gdn_aot_pack_bf16")
                .map_err(start)?,
            kv_to_vk: module
                .load_function("infinity_qwen35_gdn_aot_state_kv_to_vk")
                .map_err(start)?,
            vk_to_kv: module
                .load_function("infinity_qwen35_gdn_aot_state_vk_to_kv")
                .map_err(start)?,
        })
    }
}

/// Cold AOT owner: one provider, the packing module, and every packed/state
/// buffer for the prepared capacity. All of it stays resident for the life of
/// the prepared operator and is never reallocated per request.
struct GdnAotWork {
    provider: Arc<GdnAotProvider>,
    kernels: GdnAotKernelSet,
    packed_q: DeviceBuffer<bf16>,
    packed_k: DeviceBuffer<bf16>,
    packed_v: DeviceBuffer<bf16>,
    alpha: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    state_vk: DeviceBuffer<f32>,
    initial_vk: DeviceBuffer<f32>,
    workspace: DeviceBuffer<u8>,
}

/// Typed owners for the AOT half of a cold prepared operator. Every buffer is
/// stored here before its fallible zero enqueue, so a partial failure leaves all
/// uncertain GPU/module/provider owners under the enclosing construction guard.
struct GdnAotBuilding {
    provider: Option<Arc<GdnAotProvider>>,
    kernels: Option<GdnAotKernelSet>,
    packed_q: Option<DeviceBuffer<bf16>>,
    packed_k: Option<DeviceBuffer<bf16>>,
    packed_v: Option<DeviceBuffer<bf16>>,
    alpha: Option<DeviceBuffer<f32>>,
    beta: Option<DeviceBuffer<f32>>,
    state_vk: Option<DeviceBuffer<f32>>,
    initial_vk: Option<DeviceBuffer<f32>>,
    workspace: Option<DeviceBuffer<u8>>,
}

impl GdnAotBuilding {
    fn new() -> Self {
        Self {
            provider: None,
            kernels: None,
            packed_q: None,
            packed_k: None,
            packed_v: None,
            alpha: None,
            beta: None,
            state_vk: None,
            initial_vk: None,
            workspace: None,
        }
    }

    fn finish(self) -> Result<Option<GdnAotWork>, BackendError> {
        if self.provider.is_none() && self.kernels.is_none() {
            return Ok(None);
        }
        let missing = || start("GDN AOT cold construction lost an owner");
        Ok(Some(GdnAotWork {
            provider: self.provider.ok_or_else(missing)?,
            kernels: self.kernels.ok_or_else(missing)?,
            packed_q: self.packed_q.ok_or_else(missing)?,
            packed_k: self.packed_k.ok_or_else(missing)?,
            packed_v: self.packed_v.ok_or_else(missing)?,
            alpha: self.alpha.ok_or_else(missing)?,
            beta: self.beta.ok_or_else(missing)?,
            state_vk: self.state_vk.ok_or_else(missing)?,
            initial_vk: self.initial_vk.ok_or_else(missing)?,
            workspace: self.workspace.ok_or_else(missing)?,
        }))
    }
}

/// Cold immutable GDN weights, reusable across Work capacities. The model
/// owns one binding per layer while each Work owns a single scratch pipeline.
pub struct GdnDeviceWeights {
    context: Arc<Context>,
    geometry: GdnGeometry,
    conv_weight: DeviceBuffer<bf16>,
    norm_weights: DeviceBuffer<bf16>,
    a_log: DeviceBuffer<f32>,
    dt_bias: DeviceBuffer<f32>,
}

impl GdnDeviceWeights {
    pub fn upload(
        stream: &Arc<Stream>,
        geometry: GdnGeometry,
        weights: GdnWeights<'_>,
    ) -> Result<Self, BackendError> {
        validate_weights(geometry, &weights)?;
        Ok(Self {
            context: stream.context().clone(),
            geometry,
            conv_weight: DeviceBuffer::from_host(stream, weights.conv).map_err(start)?,
            norm_weights: DeviceBuffer::from_host(stream, weights.norm).map_err(start)?,
            a_log: DeviceBuffer::from_host(stream, weights.a_log).map_err(start)?,
            dt_bias: DeviceBuffer::from_host(stream, weights.dt_bias).map_err(start)?,
        })
    }

    pub const fn geometry(&self) -> GdnGeometry {
        self.geometry
    }

    fn validate_binding(
        &self,
        geometry: GdnGeometry,
        context: &Arc<Context>,
    ) -> Result<(), BackendError> {
        if self.geometry != geometry || !Arc::ptr_eq(&self.context, context) {
            return Err(infer(
                "layer weights have a different GDN geometry or CUDA context",
            ));
        }
        Ok(())
    }
}

struct GdnCore {
    stream: Arc<Stream>,
    kernels: GdnKernelSet,
    default_weights: Arc<GdnDeviceWeights>,
    convolution: DeviceBuffer<bf16>,
    qk: DeviceBuffer<f32>,
    decay_beta: DeviceBuffer<f32>,
    recurrence: DeviceBuffer<bf16>,
    output: DeviceBuffer<bf16>,
    final_state: DeviceBuffer<f32>,
    cu: DeviceBuffer<u32>,
    sources: DeviceBuffer<i64>,
    history_cu: DeviceBuffer<i32>,
    host_cu: Vec<u32>,
    host_sources: Vec<i64>,
    host_history_cu: Vec<i32>,
    metadata_generation: u64,
    retained_storage: Vec<RawDeviceBuffer>,
    aot: Option<GdnAotWork>,
}

/// Typed owners for one cold core. The raw module, weights, every device buffer,
/// host metadata and retention slots are retained here before any zero enqueue.
struct GdnCoreBuilding {
    stream: Arc<Stream>,
    kernels: Option<GdnKernelSet>,
    default_weights: Option<Arc<GdnDeviceWeights>>,
    convolution: Option<DeviceBuffer<bf16>>,
    qk: Option<DeviceBuffer<f32>>,
    decay_beta: Option<DeviceBuffer<f32>>,
    recurrence: Option<DeviceBuffer<bf16>>,
    output: Option<DeviceBuffer<bf16>>,
    final_state: Option<DeviceBuffer<f32>>,
    cu: Option<DeviceBuffer<u32>>,
    sources: Option<DeviceBuffer<i64>>,
    history_cu: Option<DeviceBuffer<i32>>,
    host_cu: Option<Vec<u32>>,
    host_sources: Option<Vec<i64>>,
    host_history_cu: Option<Vec<i32>>,
    metadata_generation: u64,
    retained_storage: Option<Vec<RawDeviceBuffer>>,
    aot: GdnAotBuilding,
}

impl GdnCoreBuilding {
    fn new(stream: &Arc<Stream>) -> Self {
        Self {
            stream: stream.clone(),
            kernels: None,
            default_weights: None,
            convolution: None,
            qk: None,
            decay_beta: None,
            recurrence: None,
            output: None,
            final_state: None,
            cu: None,
            sources: None,
            history_cu: None,
            host_cu: None,
            host_sources: None,
            host_history_cu: None,
            metadata_generation: 0,
            retained_storage: None,
            aot: GdnAotBuilding::new(),
        }
    }

    fn finish(self) -> Result<GdnCore, BackendError> {
        let GdnCoreBuilding {
            stream,
            kernels,
            default_weights,
            convolution,
            qk,
            decay_beta,
            recurrence,
            output,
            final_state,
            cu,
            sources,
            history_cu,
            host_cu,
            host_sources,
            host_history_cu,
            metadata_generation,
            retained_storage,
            aot,
        } = self;
        let missing = || start("GDN cold core construction lost an owner");
        Ok(GdnCore {
            stream,
            kernels: kernels.ok_or_else(missing)?,
            default_weights: default_weights.ok_or_else(missing)?,
            convolution: convolution.ok_or_else(missing)?,
            qk: qk.ok_or_else(missing)?,
            decay_beta: decay_beta.ok_or_else(missing)?,
            recurrence: recurrence.ok_or_else(missing)?,
            output: output.ok_or_else(missing)?,
            final_state: final_state.ok_or_else(missing)?,
            cu: cu.ok_or_else(missing)?,
            sources: sources.ok_or_else(missing)?,
            history_cu: history_cu.ok_or_else(missing)?,
            host_cu: host_cu.ok_or_else(missing)?,
            host_sources: host_sources.ok_or_else(missing)?,
            host_history_cu: host_history_cu.ok_or_else(missing)?,
            metadata_generation,
            retained_storage: retained_storage.ok_or_else(missing)?,
            aot: aot.finish()?,
        })
    }
}

fn build_core(
    stream: &Arc<Stream>,
    kernels: GdnKernelSet,
    geometry: GdnGeometry,
    capacity: GdnCapacity,
    weights: &GdnWeights<'_>,
    epsilon: f32,
) -> Result<Box<GdnCore>, BackendError> {
    if !Arc::ptr_eq(stream.context(), &kernels.context) {
        return Err(start("kernel module belongs to a different CUDA context"));
    }
    validate_epsilon(epsilon)?;
    let _ = GdnScratchLayout::new(geometry, capacity)?;
    validate_weights(geometry, weights)?;

    let device_weights = Arc::new(GdnDeviceWeights::upload(
        stream,
        geometry,
        GdnWeights {
            conv: weights.conv,
            a_log: weights.a_log,
            dt_bias: weights.dt_bias,
            norm: weights.norm,
        },
    )?);
    build_core_with_device_weights(stream, kernels, geometry, capacity, device_weights, epsilon)
}

/// Ordered core allocation used by both the standalone constructor and the
/// enclosing Q35 Work. It never fences: every allocation and zero enqueue is
/// retained in `building`, and the caller takes the one publication fence.
fn build_core_with_device_weights_into(
    building: &mut GdnPreparedBuilding,
    kernels: GdnKernelSet,
    geometry: GdnGeometry,
    capacity: GdnCapacity,
    weights: Arc<GdnDeviceWeights>,
    epsilon: f32,
) -> Result<(), BackendError> {
    let stream = building.stream.clone();
    if !Arc::ptr_eq(stream.context(), &kernels.context) {
        return Err(start("kernel module belongs to a different CUDA context"));
    }
    let core = building.core_mut()?;
    // Retain the module and weight owners before any other fallible step.
    core.kernels = Some(kernels);
    core.default_weights = Some(weights);
    validate_epsilon(epsilon)?;
    core.default_weights
        .as_ref()
        .ok_or_else(|| start("GDN cold core lost its weights"))?
        .validate_binding(geometry, stream.context())?;
    let layout = GdnScratchLayout::new(geometry, capacity)?;
    allocate_zero_ordered(
        &stream,
        layout.convolution_elements(),
        &mut core.convolution,
    )?;
    allocate_zero_ordered(&stream, layout.qk_elements(), &mut core.qk)?;
    allocate_zero_ordered(&stream, layout.decay_beta_elements(), &mut core.decay_beta)?;
    allocate_zero_ordered(&stream, layout.recurrence_elements(), &mut core.recurrence)?;
    allocate_zero_ordered(&stream, layout.output_elements(), &mut core.output)?;
    allocate_zero_ordered(
        &stream,
        layout.final_state_elements(),
        &mut core.final_state,
    )?;
    allocate_zero_ordered(&stream, layout.cumulative_elements(), &mut core.cu)?;
    allocate_zero_ordered(&stream, layout.source_elements(), &mut core.sources)?;
    allocate_zero_ordered(
        &stream,
        layout.history_cumulative_elements(),
        &mut core.history_cu,
    )?;
    core.host_cu = Some(zeroed_host(layout.cumulative_elements(), "cu")?);
    core.host_sources = Some(zeroed_host(layout.source_elements(), "sources")?);
    core.host_history_cu = Some(zeroed_host(
        layout.history_cumulative_elements(),
        "history_cu",
    )?);
    core.metadata_generation = 0;
    core.retained_storage = Some(retained_storage_slots()?);
    Ok(())
}

fn build_core_with_device_weights(
    stream: &Arc<Stream>,
    kernels: GdnKernelSet,
    geometry: GdnGeometry,
    capacity: GdnCapacity,
    weights: Arc<GdnDeviceWeights>,
    epsilon: f32,
) -> Result<Box<GdnCore>, BackendError> {
    let mut building = GdnPreparedBuilding::new(stream, geometry, capacity, epsilon);
    if let Err(error) = build_core_with_device_weights_into(
        &mut building,
        kernels,
        geometry,
        capacity,
        weights,
        epsilon,
    ) {
        return Err(settle_cold_error(building, stream, error));
    }
    match stream.synchronize() {
        Ok(()) => building.finish()?.into_core(),
        Err(fence_error) => {
            let _ = ColdPending::new(building, None).quarantine();
            Err(start(format!(
                "GDN cold publication fence failed: {fence_error}"
            )))
        }
    }
}

fn settle_cold_error(
    building: GdnPreparedBuilding,
    stream: &Arc<Stream>,
    original: BackendError,
) -> BackendError {
    match stream.synchronize() {
        Ok(()) => {
            drop(building);
            original
        }
        Err(fence_error) => {
            let _ = ColdPending::new(building, None).quarantine();
            start(format!(
                "{original}; GDN cold initialization fence failed: {fence_error}"
            ))
        }
    }
}

/// Cold owner of the immutable weights, reusable buffers, and exact exports.
pub struct GdnPrepared {
    stream: Arc<Stream>,
    geometry: GdnGeometry,
    capacity: GdnCapacity,
    epsilon: f32,
    core: Option<Box<GdnCore>>,
}

/// Engine-only ordered construction state. Public constructors fence this
/// owner set before returning; the enclosing Q35 Work keeps it unfenced and
/// takes the single publication fence after every zero enqueue.
pub(crate) struct GdnPreparedBuilding {
    stream: Arc<Stream>,
    geometry: GdnGeometry,
    capacity: GdnCapacity,
    epsilon: f32,
    core: Option<GdnCoreBuilding>,
}

impl GdnPreparedBuilding {
    pub(crate) fn new(
        stream: &Arc<Stream>,
        geometry: GdnGeometry,
        capacity: GdnCapacity,
        epsilon: f32,
    ) -> Self {
        Self {
            stream: stream.clone(),
            geometry,
            capacity,
            epsilon,
            core: Some(GdnCoreBuilding::new(stream)),
        }
    }

    fn core_mut(&mut self) -> Result<&mut GdnCoreBuilding, BackendError> {
        self.core
            .as_mut()
            .ok_or_else(|| start("GDN cold builder lost its core"))
    }

    pub(crate) fn finish(self) -> Result<GdnPrepared, BackendError> {
        let Self {
            stream,
            geometry,
            capacity,
            epsilon,
            core,
        } = self;
        let core = core
            .ok_or_else(|| start("GDN cold builder lost its core"))?
            .finish()?;
        Ok(GdnPrepared {
            stream,
            geometry,
            capacity,
            epsilon,
            core: Some(Box::new(core)),
        })
    }
}

impl GdnPrepared {
    fn into_core(self) -> Result<Box<GdnCore>, BackendError> {
        let GdnPrepared { core, .. } = self;
        core.ok_or_else(|| start("GDN prepared construction lost its core"))
    }

    /// Cold construction: validate geometry, capacity, epsilon, and weight
    /// lengths, then upload weights and allocate all reusable scratch once.
    pub fn new(
        stream: &Arc<Stream>,
        kernels: GdnKernelSet,
        geometry: GdnGeometry,
        capacity: GdnCapacity,
        weights: GdnWeights<'_>,
        epsilon: f32,
    ) -> Result<Self, BackendError> {
        let core = build_core(stream, kernels, geometry, capacity, &weights, epsilon)?;
        Ok(Self {
            stream: stream.clone(),
            geometry,
            capacity,
            epsilon,
            core: Some(core),
        })
    }

    /// Cold construction of the default dense pipeline: the same admitted raw
    /// weights, convolution, prepare, norm, retention, and quarantine paths plus
    /// the upstream AOT provider and the separate packing module.
    #[allow(clippy::too_many_arguments)]
    pub fn new_default_dense(
        stream: &Arc<Stream>,
        kernels: GdnKernelSet,
        geometry: GdnGeometry,
        capacity: GdnCapacity,
        weights: GdnWeights<'_>,
        epsilon: f32,
        provider_root: &Path,
        aux_bundle: &Path,
    ) -> Result<Self, BackendError> {
        if !Arc::ptr_eq(stream.context(), &kernels.context) {
            return Err(start("kernel module belongs to a different CUDA context"));
        }
        validate_epsilon(epsilon)?;
        let _ = GdnScratchLayout::new(geometry, capacity)?;
        let weights = Arc::new(GdnDeviceWeights::upload(stream, geometry, weights)?);
        Self::new_default_dense_with_device_weights(
            stream,
            kernels,
            capacity,
            weights,
            epsilon,
            provider_root,
            aux_bundle,
        )
    }

    /// Reuse cold model weights while allocating only one scratch pipeline per Work.
    pub fn new_default_dense_with_device_weights(
        stream: &Arc<Stream>,
        kernels: GdnKernelSet,
        capacity: GdnCapacity,
        weights: Arc<GdnDeviceWeights>,
        epsilon: f32,
        provider_root: &Path,
        aux_bundle: &Path,
    ) -> Result<Self, BackendError> {
        let geometry = weights.geometry;
        if !Arc::ptr_eq(stream.context(), &kernels.context) {
            return Err(start("kernel module belongs to a different CUDA context"));
        }
        validate_epsilon(epsilon)?;
        weights.validate_binding(geometry, stream.context())?;
        let _ = GdnScratchLayout::new(geometry, capacity)?;
        let mut building = GdnPreparedBuilding::new(stream, geometry, capacity, epsilon);
        if let Err(error) = Self::build_default_dense_into(
            &mut building,
            stream,
            kernels,
            capacity,
            weights,
            epsilon,
            provider_root,
            aux_bundle,
        ) {
            return Err(settle_cold_error(building, stream, error));
        }
        match stream.synchronize() {
            Ok(()) => building.finish(),
            Err(fence_error) => {
                let _ = ColdPending::new(building, None).quarantine();
                Err(start(format!(
                    "GDN cold publication fence failed: {fence_error}"
                )))
            }
        }
    }

    /// Engine-only ordered default-dense construction. It allocates and zeroes
    /// every owner into `building` without a fence; the caller owns the single
    /// publication fence and the error-path release/quarantine decision.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_default_dense_into(
        building: &mut GdnPreparedBuilding,
        stream: &Arc<Stream>,
        kernels: GdnKernelSet,
        capacity: GdnCapacity,
        weights: Arc<GdnDeviceWeights>,
        epsilon: f32,
        provider_root: &Path,
        aux_bundle: &Path,
    ) -> Result<(), BackendError> {
        let geometry = weights.geometry;
        if !Arc::ptr_eq(stream, &building.stream) {
            return Err(start("GDN cold builder stream changed"));
        }
        if !Arc::ptr_eq(stream.context(), &kernels.context) {
            return Err(start("kernel module belongs to a different CUDA context"));
        }
        validate_epsilon(epsilon)?;
        weights.validate_binding(geometry, stream.context())?;
        if geometry != building.geometry || capacity != building.capacity {
            return Err(start("GDN cold builder binding changed"));
        }
        if epsilon != building.epsilon {
            return Err(start("GDN cold builder epsilon changed"));
        }
        let _ = GdnScratchLayout::new(geometry, capacity)?;
        let context = kernels.context.clone();
        let validated_aux = GdnAotValidatedBundle::load(aux_bundle)?;
        let provider = Arc::new(GdnAotProvider::load(provider_root, context.clone())?);
        let aot_kernels = GdnAotKernelSet::bind(&context, &validated_aux)?;
        let aot_layout = GdnAotLayout::new(geometry, capacity)?;
        let workspace_bytes = usize::try_from(provider.workspace_bytes())
            .map_err(|_| start("AOT provider workspace exceeds the host allocation range"))?;
        if workspace_bytes == 0 {
            return Err(start("AOT provider workspace must be positive"));
        }
        // Retain the provider and its module before the first fallible zero.
        {
            let core = building.core_mut()?;
            core.aot.provider = Some(provider);
            core.aot.kernels = Some(aot_kernels);
        }
        build_core_with_device_weights_into(
            building, kernels, geometry, capacity, weights, epsilon,
        )?;
        let stream = building.stream.clone();
        let core = building.core_mut()?;
        allocate_zero_ordered(&stream, aot_layout.packed_qk, &mut core.aot.packed_q)?;
        allocate_zero_ordered(&stream, aot_layout.packed_qk, &mut core.aot.packed_k)?;
        allocate_zero_ordered(&stream, aot_layout.packed_v, &mut core.aot.packed_v)?;
        allocate_zero_ordered(&stream, aot_layout.gates, &mut core.aot.alpha)?;
        allocate_zero_ordered(&stream, aot_layout.gates, &mut core.aot.beta)?;
        allocate_zero_ordered(&stream, aot_layout.state, &mut core.aot.state_vk)?;
        allocate_zero_ordered(&stream, aot_layout.state, &mut core.aot.initial_vk)?;
        allocate_zero_ordered(&stream, workspace_bytes, &mut core.aot.workspace)?;
        Ok(())
    }

    fn core_ref(&self) -> Result<&GdnCore, BackendError> {
        let core = self.core.as_deref().ok_or(BackendError::Unhealthy)?;
        if core
            .aot
            .as_ref()
            .is_some_and(|work| !work.provider.is_healthy())
        {
            return Err(BackendError::Unhealthy);
        }
        Ok(core)
    }

    fn core_mut(&mut self) -> Result<&mut GdnCore, BackendError> {
        self.core.as_deref_mut().ok_or(BackendError::Unhealthy)
    }

    pub const fn geometry(&self) -> GdnGeometry {
        self.geometry
    }

    pub const fn capacity(&self) -> GdnCapacity {
        self.capacity
    }

    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Enqueue in an engine-owned arena without introducing a per-layer drain.
    ///
    /// # Safety
    /// The outer Work must own this Prepared, stream, EVERY borrowed per-layer
    /// GdnDeviceWeights binding (all four buffers), every input/history
    /// allocation, and their immutable metadata through its completion event.
    /// Its context is already bound on this thread. No owner may be reused,
    /// mutated or evicted while any queued work can access it. On ANY error,
    /// the outer Work must drain its stream or quarantine ALL its owners;
    /// `queued` describes this GDN call, not earlier work on the same stream.
    /// Successful return does not synchronize and the returned views do not
    /// implement a completion/draining Drop.
    #[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 engine integration.
    pub(crate) unsafe fn enqueue_owned<'a>(
        &'a mut self,
        inputs: GdnInputViews<'_>,
        cumulative: &[u32],
        state: GdnState<'_>,
        layer_weights: &GdnDeviceWeights,
        write_final_state: bool,
    ) -> Result<GdnOwnedOutput<'a>, GdnOwnedFailure> {
        let early = |error| GdnOwnedFailure {
            error,
            queued: false,
        };
        layer_weights
            .validate_binding(self.geometry, self.stream.context())
            .map_err(early)?;
        let plan = self
            .preflight(&inputs, cumulative, &state, write_final_state)
            .map_err(early)?;
        stage_host_metadata(self.core_mut().map_err(early)?, cumulative, &state).map_err(early)?;
        enqueue(
            self.core_ref().map_err(early)?,
            &plan,
            &inputs,
            layer_weights,
        )
        .map_err(|failure| GdnOwnedFailure {
            error: failure.error,
            queued: failure.queued,
        })?;
        let late = |error| GdnOwnedFailure {
            error,
            queued: true,
        };
        let core = self.core_ref().map_err(late)?;
        let output = GdnBufferView::from_buffer_range(&core.output, 0..plan.output_elements)
            .map_err(late)?;
        let final_state = match plan.final_elements {
            Some(elements) => Some(
                GdnBufferView::from_buffer_range(&core.final_state, 0..elements).map_err(late)?,
            ),
            None => None,
        };
        Ok(GdnOwnedOutput {
            output,
            final_state,
        })
    }

    /// Validate the request, bind the prepared context, retain every borrowed
    /// owner before the first enqueue, and launch the exact native sequence.
    pub fn submit<'a>(
        &'a mut self,
        inputs: GdnInputs<'a>,
        cumulative: &[u32],
        state: GdnState<'a>,
        write_final_state: bool,
    ) -> Result<GdnSubmission<'a>, BackendError> {
        let borrowed = inputs.borrowed();
        let plan = self.preflight(&borrowed, cumulative, &state, write_final_state)?;

        self.stream.context().bind_to_thread().map_err(start)?;

        let mut retained = {
            let core = self.core_mut()?;
            retain_inputs(core, &borrowed, &state, plan.retention_slots)?
        };

        let stage = match self.core_mut() {
            Ok(core) => stage_host_metadata(core, cumulative, &state),
            Err(error) => Err(error),
        };
        if let Err(error) = stage {
            retained.clear();
            if let Some(core) = self.core.as_mut() {
                core.retained_storage = retained;
            }
            return Err(error);
        }

        let enqueue = match self.core_ref() {
            Ok(core) => enqueue(core, &plan, &borrowed, &core.default_weights),
            Err(error) => Err(GdnEnqueueFailure {
                error,
                queued: false,
            }),
        };
        match enqueue {
            Ok(()) => Ok(GdnSubmission {
                prepared: self,
                _inputs: inputs,
                _state: state,
                output_elements: plan.output_elements,
                final_elements: plan.final_elements,
                retained,
                completed: false,
            }),
            Err(failure) => {
                if failure.queued {
                    let drain = self.stream.synchronize().map_err(infer);
                    match settle_pending(&mut self.core, &mut retained, drain) {
                        PendingDrain::Drained => {
                            retained.clear();
                            if let Some(core) = self.core.as_mut() {
                                core.retained_storage = retained;
                            }
                            Err(failure.error)
                        }
                        PendingDrain::Quarantined { error } => Err(infer(format!(
                            "{}; the prepared operator is poisoned after a failed drain: {error}",
                            failure.error
                        ))),
                    }
                } else {
                    retained.clear();
                    if let Some(core) = self.core.as_mut() {
                        core.retained_storage = retained;
                    }
                    Err(failure.error)
                }
            }
        }
    }

    fn preflight(
        &self,
        inputs: &GdnInputViews<'_>,
        cumulative: &[u32],
        state: &GdnState<'_>,
        write_final_state: bool,
    ) -> Result<GdnSubmitPlan, BackendError> {
        self.preflight_views(inputs, cumulative, state.borrowed(), write_final_state)
    }

    fn preflight_views(
        &self,
        inputs: &GdnInputViews<'_>,
        cumulative: &[u32],
        state: GdnStateViews<'_>,
        write_final_state: bool,
    ) -> Result<GdnSubmitPlan, BackendError> {
        let core = self.core_ref()?;
        let plan = GdnPlan::new(self.geometry, self.capacity, cumulative)?;
        let layout = GdnInputLayout::admit(
            self.geometry,
            plan.tokens,
            inputs.qkv_extent(),
            inputs.z_extent(),
            inputs.ab_extent(),
        )?;
        require_context(&core.stream, inputs.qkv.context(), "qkv")?;
        require_context(&core.stream, inputs.z.context(), "z")?;
        require_context(&core.stream, inputs.ab.context(), "a/b")?;

        let (initial_pointer, history_pointer, cached_source_count) = match state {
            GdnStateViews::Fresh => (0u64, 0u64, 0usize),
            GdnStateViews::Cached {
                initial,
                history_rows,
                history_cumulative,
                sources,
            } => {
                require_context(&core.stream, initial.context(), "initial state")?;
                require_context(&core.stream, history_rows.context(), "convolution history")?;
                let rows = history_row_count(history_rows.len(), self.geometry.channels())?;
                let history_plan = GdnHistoryPlan::new(
                    self.geometry,
                    plan.sequences,
                    sources,
                    history_cumulative,
                    rows,
                )?;
                if initial.len() != history_plan.initial_elements() {
                    return Err(infer(format!(
                        "initial state has {} elements, expected {}",
                        initial.len(),
                        history_plan.initial_elements()
                    )));
                }
                if history_rows.len() != history_plan.history_elements() {
                    return Err(infer(format!(
                        "convolution history has {} elements, expected {}",
                        history_rows.len(),
                        history_plan.history_elements()
                    )));
                }
                if history_plan.initial_sequences() > self.capacity.history_sequences {
                    return Err(infer(format!(
                        "history sources {} exceed prepared capacity {}",
                        history_plan.initial_sequences(),
                        self.capacity.history_sequences
                    )));
                }
                if history_plan.history_rows() > GDN_MAX_TOKENS {
                    return Err(infer(
                        "cached history row count exceeds the native int32 bound",
                    ));
                }
                if pointer_alignment(initial.cu_deviceptr()) < 4 {
                    return Err(infer("initial state base alignment is below four bytes"));
                }
                if history_plan.history_elements() != 0
                    && pointer_alignment(history_rows.cu_deviceptr()) < 8
                {
                    return Err(infer(
                        "convolution history base alignment is below eight bytes",
                    ));
                }
                (
                    initial.cu_deviceptr(),
                    history_rows.cu_deviceptr(),
                    history_plan.initial_sequences(),
                )
            }
        };

        let extents = request_extents(self.geometry, plan, write_final_state)?;
        if extents.output_elements > core.output.len() {
            return Err(infer(format!(
                "request output {} exceeds prepared capacity {}",
                extents.output_elements,
                core.output.len()
            )));
        }
        if let Some(elements) = extents.final_elements {
            if elements > core.final_state.len() {
                return Err(infer(format!(
                    "request final state {elements} exceeds prepared capacity {}",
                    core.final_state.len()
                )));
            }
        }

        let cached = matches!(state, GdnStateViews::Cached { .. });
        let aot = match core.aot.as_ref() {
            Some(work) if aot_admits(cached, plan.tokens, plan.sequences) => Some(aot_submit_plan(
                work,
                core,
                self.geometry,
                plan,
                initial_pointer,
                cached,
                cached_source_count,
                write_final_state,
            )?),
            _ => None,
        };

        if cumulative.len() > core.host_cu.capacity() {
            return Err(infer(format!(
                "cumulative sequence metadata has {} entries, prepared capacity is {}",
                cumulative.len(),
                core.host_cu.capacity()
            )));
        }
        let mut retention_slots = 3;
        if let GdnStateViews::Cached {
            history_cumulative,
            sources,
            ..
        } = state
        {
            let lengths = cached_metadata_lengths(history_cumulative, sources);
            if lengths.cumulative > core.host_history_cu.capacity() {
                return Err(infer(format!(
                    "cached history cumulative metadata has {} entries, prepared capacity is {}",
                    lengths.cumulative,
                    core.host_history_cu.capacity()
                )));
            }
            if lengths.sources > core.host_sources.capacity() {
                return Err(infer(format!(
                    "cached source map has {} entries, prepared capacity is {}",
                    lengths.sources,
                    core.host_sources.capacity()
                )));
            }
            retention_slots += 2;
        }
        if core.retained_storage.capacity() < retention_slots {
            return Err(start("input allocation retention slots are unavailable"));
        }

        let cached_lanes = if matches!(state, GdnStateViews::Cached { .. }) || write_final_state {
            Some(self.geometry.cached_lanes(plan.sequences)?)
        } else {
            None
        };
        let packed_grid = self.geometry.packed_conv_grid(plan.tokens)?;
        let prepare_grid = (
            grid_u32(
                product(plan.tokens, self.geometry.key_heads, "prepare grid")?,
                "prepare grid",
            )?,
            1,
            1,
        );
        let norm_grid = (
            grid_u32(
                product(plan.tokens, self.geometry.value_heads, "norm grid")?,
                "norm grid",
            )?,
            1,
            1,
        );
        let sequence_grid = grid_u32(plan.sequences, "recurrence sequences")?;
        let head_grid = grid_u32(self.geometry.value_heads, "recurrence heads")?;
        let recurrence_grid = match cached_lanes {
            Some(lanes) => (sequence_grid, head_grid, lanes / 4),
            None => (sequence_grid, head_grid, 16),
        };

        let channels = int32_value(self.geometry.channels(), "channels")?;
        let kernel = int32_value(self.geometry.kernel, "kernel")?;
        let sequences = int32_value(plan.sequences, "sequences")?;
        let qkv_stride = i64::try_from(layout.qkv_row_stride)
            .map_err(|_| infer("qkv row stride exceeds int64"))?;
        let tokens = int32_value(plan.tokens, "tokens")?;
        let key_heads = int32_value(self.geometry.key_heads, "key heads")?;
        let value_heads = int32_value(self.geometry.value_heads, "value heads")?;
        let cu_bytes = metadata_bytes::<u32>(cumulative.len())?;
        let (history_cu_bytes, sources_bytes) = match state {
            GdnStateViews::Fresh => (0, 0),
            GdnStateViews::Cached {
                history_cumulative,
                sources,
                ..
            } => (
                metadata_bytes::<i32>(history_cumulative.len())?,
                metadata_bytes::<i64>(sources.len())?,
            ),
        };

        Ok(GdnSubmitPlan {
            tokens,
            sequences,
            channels,
            kernel,
            key_heads,
            value_heads,
            qkv_stride,
            initial_pointer,
            history_pointer,
            use_cached_convolution: matches!(state, GdnStateViews::Cached { .. }),
            cached_lanes,
            packed_grid,
            prepare_grid,
            norm_grid,
            recurrence_grid,
            epsilon: self.epsilon,
            output_elements: extents.output_elements,
            final_elements: extents.final_elements,
            retention_slots,
            cu_bytes,
            history_cu_bytes,
            sources_bytes,
            aot,
        })
    }
}

struct GdnSubmitPlan {
    tokens: i32,
    sequences: i32,
    channels: i32,
    kernel: i32,
    key_heads: i32,
    value_heads: i32,
    qkv_stride: i64,
    initial_pointer: u64,
    history_pointer: u64,
    use_cached_convolution: bool,
    cached_lanes: Option<u32>,
    packed_grid: (u32, u32, u32),
    prepare_grid: (u32, u32, u32),
    norm_grid: (u32, u32, u32),
    recurrence_grid: (u32, u32, u32),
    epsilon: f32,
    output_elements: usize,
    final_elements: Option<usize>,
    retention_slots: usize,
    cu_bytes: usize,
    history_cu_bytes: usize,
    sources_bytes: usize,
    aot: Option<GdnAotSubmitPlan>,
}

struct GdnAotSubmitPlan {
    pack_grid: (u32, u32, u32),
    state_grid: (u32, u32, u32),
    cached: bool,
    cached_sources: usize,
    write_final: bool,
    workspace_bytes: u64,
}

/// Pure request admission for the cold AOT owner. Every provider/source shape,
/// capacity, pointer and alignment fact is checked here, before the first
/// enqueue, so the hot enqueue performs no CUDA query or allocation.
#[allow(clippy::too_many_arguments)]
fn aot_submit_plan(
    work: &GdnAotWork,
    core: &GdnCore,
    geometry: GdnGeometry,
    plan: GdnPlan,
    initial_pointer: u64,
    cached: bool,
    cached_sources: usize,
    write_final: bool,
) -> Result<GdnAotSubmitPlan, BackendError> {
    let packed_qk = product(
        product(plan.tokens, geometry.key_heads, "AOT request Q/K rows")?,
        GDN_KEY_DIM,
        "AOT request Q/K",
    )?;
    if packed_qk > work.packed_q.len() || packed_qk > work.packed_k.len() {
        return Err(infer(format!(
            "AOT request Q/K {packed_qk} exceeds prepared capacity {}",
            work.packed_q.len().min(work.packed_k.len())
        )));
    }
    let packed_v = product(
        product(plan.tokens, geometry.value_heads, "AOT request V rows")?,
        GDN_KEY_DIM,
        "AOT request V",
    )?;
    if packed_v > work.packed_v.len() {
        return Err(infer(format!(
            "AOT request V {packed_v} exceeds prepared capacity {}",
            work.packed_v.len()
        )));
    }
    let gates = product(plan.tokens, geometry.value_heads, "AOT request gates")?;
    if gates > work.alpha.len() || gates > work.beta.len() {
        return Err(infer(format!(
            "AOT request gates {gates} exceed prepared capacity {}",
            work.alpha.len().min(work.beta.len())
        )));
    }
    let state = product(
        product(
            plan.sequences,
            geometry.value_heads,
            "AOT request state heads",
        )?,
        geometry.state_row_width(),
        "AOT request state",
    )?;
    if state > work.state_vk.len() {
        return Err(infer(format!(
            "AOT request state {state} exceeds prepared capacity {}",
            work.state_vk.len()
        )));
    }
    if cached {
        if work.initial_vk.len() < state {
            return Err(infer(format!(
                "AOT cached state {state} exceeds prepared capacity {}",
                work.initial_vk.len()
            )));
        }
        if cached_sources == 0 {
            return Err(infer("AOT cached request has no initial-state sources"));
        }
    }
    let workspace_bytes = u64::try_from(work.workspace.len())
        .map_err(|_| infer("AOT workspace byte extent exceeds u64"))?;
    if workspace_bytes != u64::try_from(work.provider.workspace_bytes()).unwrap_or(u64::MAX) {
        return Err(infer("AOT provider workspace plan is inconsistent"));
    }
    require_aot_alignment(work.packed_q.cu_deviceptr(), 16, "AOT packed Q")?;
    require_aot_alignment(work.packed_k.cu_deviceptr(), 16, "AOT packed K")?;
    require_aot_alignment(work.packed_v.cu_deviceptr(), 16, "AOT packed V")?;
    require_aot_alignment(work.alpha.cu_deviceptr(), 16, "AOT alpha")?;
    require_aot_alignment(work.beta.cu_deviceptr(), 16, "AOT beta")?;
    require_aot_alignment(work.state_vk.cu_deviceptr(), 16, "AOT state")?;
    require_aot_alignment(work.initial_vk.cu_deviceptr(), 16, "AOT initial state")?;
    require_aot_alignment(work.workspace.cu_deviceptr(), 128, "AOT workspace")?;
    require_aot_alignment(core.recurrence.cu_deviceptr(), 16, "AOT output")?;
    let pointers = checked_aot_cumulative(core, work, core.cu.cu_deviceptr())?;
    if cached {
        require_aot_alignment(initial_pointer, 16, "AOT initial source")?;
        if pointers.contains(&initial_pointer) {
            return Err(infer("AOT initial state must not alias another buffer"));
        }
    }
    let pack_grid = aot_pack_grid(plan.tokens, geometry.value_heads)?;
    let state_grid = (
        aot_state_tile_count(
            plan.sequences,
            geometry.value_heads,
            GDN_KEY_DIM,
            GDN_VALUE_DIM,
        )?,
        1,
        1,
    );
    Ok(GdnAotSubmitPlan {
        pack_grid,
        state_grid,
        cached,
        cached_sources,
        write_final,
        workspace_bytes,
    })
}

/// Validate the concrete cumulative binding used by either the standalone
/// core allocation or an engine-owned uploaded metadata view. Distinct live
/// allocations supply the stronger non-overlapping-span ownership guarantee.
fn checked_aot_cumulative(
    core: &GdnCore,
    work: &GdnAotWork,
    cumulative: u64,
) -> Result<[u64; 9], BackendError> {
    require_aot_alignment(cumulative, 4, "AOT cumulative")?;
    let pointers = [
        work.packed_q.cu_deviceptr(),
        work.packed_k.cu_deviceptr(),
        work.packed_v.cu_deviceptr(),
        work.alpha.cu_deviceptr(),
        work.beta.cu_deviceptr(),
        core.recurrence.cu_deviceptr(),
        cumulative,
        work.state_vk.cu_deviceptr(),
        work.workspace.cu_deviceptr(),
    ];
    if !aot_distinct(&pointers) {
        return Err(infer("AOT provider buffers must not alias"));
    }
    Ok(pointers)
}

fn aot_distinct(values: &[u64]) -> bool {
    values
        .iter()
        .enumerate()
        .all(|(index, value)| !values.iter().skip(index + 1).any(|other| other == value))
}

fn require_aot_alignment(pointer: u64, alignment: usize, name: &str) -> Result<(), BackendError> {
    if pointer == 0 || !pointer_alignment(pointer).is_multiple_of(alignment) {
        return Err(infer(format!(
            "{name} pointer is not {alignment}-byte aligned"
        )));
    }
    Ok(())
}

/// Views into the enclosing Work's prepared buffers. No synchronization on Drop.
#[allow(dead_code)] // Bound by the forthcoming crate-local engine.
pub(crate) struct GdnOwnedOutput<'a> {
    pub(crate) output: GdnBufferView<'a, bf16>,
    pub(crate) final_state: Option<GdnBufferView<'a, f32>>,
}

/// The enclosing Work owns failure drainage/quarantine, including earlier work
/// and every borrowed per-layer weight binding and input/history allocation.
#[derive(Debug)]
#[allow(dead_code)] // Bound by the forthcoming crate-local engine.
pub(crate) struct GdnOwnedFailure {
    pub(crate) error: BackendError,
    pub(crate) queued: bool,
}

struct GdnEnqueueFailure {
    error: BackendError,
    queued: bool,
}

/// Device bindings for one body. The default path supplies the prepared core
/// pointers byte-for-byte; engine block frames supply already-uploaded Work
/// metadata and an external final-state target instead.
#[derive(Clone, Copy)]
struct GdnEnqueueBindings {
    cumulative: u64,
    history_cu: u64,
    sources: u64,
    final_state: u64,
    output: u64,
}

fn enqueue(
    core: &GdnCore,
    plan: &GdnSubmitPlan,
    inputs: &GdnInputViews<'_>,
    weights: &GdnDeviceWeights,
) -> Result<(), GdnEnqueueFailure> {
    let mut queued = false;
    enqueue_metadata(core, plan, &mut queued)?;
    enqueue_body(
        core,
        plan,
        inputs,
        weights,
        core.cu.cu_deviceptr(),
        &mut queued,
    )
}

fn enqueue_metadata(
    core: &GdnCore,
    plan: &GdnSubmitPlan,
    queued: &mut bool,
) -> Result<(), GdnEnqueueFailure> {
    upload_host_async(&core.stream, &core.cu, &core.host_cu, plan.cu_bytes, queued)?;
    if plan.use_cached_convolution {
        upload_host_async(
            &core.stream,
            &core.history_cu,
            &core.host_history_cu,
            plan.history_cu_bytes,
            queued,
        )?;
        upload_host_async(
            &core.stream,
            &core.sources,
            &core.host_sources,
            plan.sources_bytes,
            queued,
        )?;
    }

    Ok(())
}

fn enqueue_body(
    core: &GdnCore,
    plan: &GdnSubmitPlan,
    inputs: &GdnInputViews<'_>,
    weights: &GdnDeviceWeights,
    cumulative: u64,
    queued: &mut bool,
) -> Result<(), GdnEnqueueFailure> {
    enqueue_body_with_bindings(
        core,
        plan,
        inputs,
        weights,
        GdnEnqueueBindings {
            cumulative,
            history_cu: core.history_cu.cu_deviceptr(),
            sources: core.sources.cu_deviceptr(),
            final_state: core.final_state.cu_deviceptr(),
            output: core.output.cu_deviceptr(),
        },
        queued,
    )
}

fn enqueue_body_with_bindings(
    core: &GdnCore,
    plan: &GdnSubmitPlan,
    inputs: &GdnInputViews<'_>,
    weights: &GdnDeviceWeights,
    bindings: GdnEnqueueBindings,
    queued: &mut bool,
) -> Result<(), GdnEnqueueFailure> {
    if plan.final_elements.is_some() && bindings.final_state == 0 {
        return Err(GdnEnqueueFailure {
            error: infer("GDN final-state write has no device target"),
            queued: *queued,
        });
    }
    let mut qkv = inputs.qkv.cu_deviceptr();
    let mut conv_weight = weights.conv_weight.cu_deviceptr();
    let mut cu = bindings.cumulative;
    let mut convolution = core.convolution.cu_deviceptr();
    let mut channels_arg = plan.channels;
    let mut kernel_arg = plan.kernel;
    let mut sequences_arg = plan.sequences;
    let mut stride_arg = plan.qkv_stride;

    if plan.use_cached_convolution {
        let mut sources = bindings.sources;
        let mut history = plan.history_pointer;
        let mut history_cu = bindings.history_cu;
        let mut args = [
            arg(&mut qkv),
            arg(&mut conv_weight),
            arg(&mut cu),
            arg(&mut convolution),
            arg(&mut channels_arg),
            arg(&mut kernel_arg),
            arg(&mut sequences_arg),
            arg(&mut stride_arg),
            arg(&mut sources),
            arg(&mut history),
            arg(&mut history_cu),
        ];
        launch_raw(
            &core.kernels.packed_conv_cached,
            &core.stream,
            plan.packed_grid,
            PACKED_CONV_THREADS,
            &mut args,
            queued,
        )?;
    } else {
        let mut args = [
            arg(&mut qkv),
            arg(&mut conv_weight),
            arg(&mut cu),
            arg(&mut convolution),
            arg(&mut channels_arg),
            arg(&mut kernel_arg),
            arg(&mut sequences_arg),
            arg(&mut stride_arg),
        ];
        launch_raw(
            &core.kernels.packed_conv_fresh,
            &core.stream,
            plan.packed_grid,
            PACKED_CONV_THREADS,
            &mut args,
            queued,
        )?;
    }

    let mut tokens_arg = plan.tokens;
    let mut kh_arg = plan.key_heads;
    let mut vh_arg = plan.value_heads;
    let mut qk = core.qk.cu_deviceptr();
    let mut decay = core.decay_beta.cu_deviceptr();
    let mut a_log = weights.a_log.cu_deviceptr();
    let mut dt_bias = weights.dt_bias.cu_deviceptr();
    let mut ab = inputs.ab.cu_deviceptr();
    let mut prepare_args = [
        arg(&mut convolution),
        arg(&mut ab),
        arg(&mut a_log),
        arg(&mut dt_bias),
        arg(&mut qk),
        arg(&mut decay),
        arg(&mut tokens_arg),
        arg(&mut kh_arg),
        arg(&mut vh_arg),
    ];
    launch_raw(
        &core.kernels.prepare,
        &core.stream,
        plan.prepare_grid,
        PREPARE_THREADS,
        &mut prepare_args,
        queued,
    )?;

    let mut recurrence = core.recurrence.cu_deviceptr();
    let mut final_pointer = if plan.final_elements.is_some() {
        bindings.final_state
    } else {
        0
    };
    if let Some(aot) = plan.aot.as_ref() {
        enqueue_aot(core, plan, aot, bindings, queued)?;
    } else if let Some(lanes) = plan.cached_lanes {
        let mut sources = if plan.initial_pointer == 0 {
            0
        } else {
            bindings.sources
        };
        let mut initial = plan.initial_pointer;
        let function = if lanes == 8 {
            &core.kernels.cached_lanes8
        } else {
            &core.kernels.cached_lanes16
        };
        let mut args = [
            arg(&mut convolution),
            arg(&mut qk),
            arg(&mut decay),
            arg(&mut cu),
            arg(&mut recurrence),
            arg(&mut kh_arg),
            arg(&mut vh_arg),
            arg(&mut initial),
            arg(&mut final_pointer),
            arg(&mut sources),
        ];
        launch_raw(
            function,
            &core.stream,
            plan.recurrence_grid,
            CACHED_THREADS,
            &mut args,
            queued,
        )?;
    } else {
        let mut args = [
            arg(&mut convolution),
            arg(&mut qk),
            arg(&mut decay),
            arg(&mut cu),
            arg(&mut recurrence),
            arg(&mut kh_arg),
            arg(&mut vh_arg),
        ];
        launch_raw(
            &core.kernels.recurrent,
            &core.stream,
            plan.recurrence_grid,
            RECURRENT_THREADS,
            &mut args,
            queued,
        )?;
    }

    let mut z = inputs.z.cu_deviceptr();
    let mut norm_weights = weights.norm_weights.cu_deviceptr();
    let mut output = bindings.output;
    let mut epsilon = plan.epsilon;
    let mut norm_args = [
        arg(&mut recurrence),
        arg(&mut z),
        arg(&mut norm_weights),
        arg(&mut output),
        arg(&mut epsilon),
    ];
    launch_raw(
        &core.kernels.norm,
        &core.stream,
        plan.norm_grid,
        NORM_THREADS,
        &mut norm_args,
        queued,
    )?;

    Ok(())
}

/// Pack the already-computed Q/K, convolution V tail and prepared A/B halves,
/// transpose/gather the cached initial state, run the exact ABI2 provider on
/// the owned stream, and transpose the final state back when requested. The
/// provider always writes its own VK state buffer.
fn enqueue_aot(
    core: &GdnCore,
    plan: &GdnSubmitPlan,
    aot: &GdnAotSubmitPlan,
    bindings: GdnEnqueueBindings,
    queued: &mut bool,
) -> Result<(), GdnEnqueueFailure> {
    let work = core.aot.as_ref().ok_or_else(|| GdnEnqueueFailure {
        error: infer("AOT work is unavailable for this submission"),
        queued: *queued,
    })?;

    let mut qk = core.qk.cu_deviceptr();
    let mut convolution = core.convolution.cu_deviceptr();
    let mut prepared_ab = core.decay_beta.cu_deviceptr();
    let mut packed_q = work.packed_q.cu_deviceptr();
    let mut packed_k = work.packed_k.cu_deviceptr();
    let mut packed_v = work.packed_v.cu_deviceptr();
    let mut alpha = work.alpha.cu_deviceptr();
    let mut beta = work.beta.cu_deviceptr();
    let mut kh_arg = plan.key_heads;
    let mut vh_arg = plan.value_heads;
    let mut qk_stride = (2 * GDN_KEY_DIM) as i64;
    let mut conv_stride = i64::from(plan.channels);
    let mut ab_stride = i64::from(plan.value_heads) * 2;
    let mut args = [
        arg(&mut qk),
        arg(&mut convolution),
        arg(&mut prepared_ab),
        arg(&mut packed_q),
        arg(&mut packed_k),
        arg(&mut packed_v),
        arg(&mut alpha),
        arg(&mut beta),
        arg(&mut kh_arg),
        arg(&mut vh_arg),
        arg(&mut qk_stride),
        arg(&mut conv_stride),
        arg(&mut ab_stride),
    ];
    launch_raw(
        &work.kernels.pack,
        &core.stream,
        aot.pack_grid,
        AOT_PACK_THREADS,
        &mut args,
        queued,
    )?;

    if aot.cached {
        let mut source = plan.initial_pointer;
        let mut destination = work.initial_vk.cu_deviceptr();
        let mut sources = bindings.sources;
        let mut batch = i64::from(plan.sequences);
        let mut source_count =
            i64::try_from(aot.cached_sources).map_err(|_| GdnEnqueueFailure {
                error: infer("AOT cached source count exceeds int64"),
                queued: *queued,
            })?;
        let mut value_heads = i64::from(plan.value_heads);
        let mut key_dim = GDN_KEY_DIM as i64;
        let mut value_dim = GDN_VALUE_DIM as i64;
        let mut state_args = [
            arg(&mut source),
            arg(&mut destination),
            arg(&mut sources),
            arg(&mut batch),
            arg(&mut source_count),
            arg(&mut value_heads),
            arg(&mut key_dim),
            arg(&mut value_dim),
        ];
        launch_aot_state(
            &work.kernels.kv_to_vk,
            &core.stream,
            aot.state_grid,
            &mut state_args,
            queued,
        )?;
    }

    let tokens = u64::try_from(plan.tokens).map_err(|_| GdnEnqueueFailure {
        error: infer("AOT token count exceeds u64"),
        queued: *queued,
    })?;
    let batch = u64::try_from(plan.sequences).map_err(|_| GdnEnqueueFailure {
        error: infer("AOT sequence count exceeds u64"),
        queued: *queued,
    })?;
    let key_heads = u64::try_from(plan.key_heads).map_err(|_| GdnEnqueueFailure {
        error: infer("AOT key head count exceeds u64"),
        queued: *queued,
    })?;
    let value_heads = u64::try_from(plan.value_heads).map_err(|_| GdnEnqueueFailure {
        error: infer("AOT value head count exceeds u64"),
        queued: *queued,
    })?;
    let q = work.packed_q.cu_deviceptr();
    let k = work.packed_k.cu_deviceptr();
    let v = work.packed_v.cu_deviceptr();
    let provider_alpha = work.alpha.cu_deviceptr();
    let provider_beta = work.beta.cu_deviceptr();
    let output = core.recurrence.cu_deviceptr();
    let state = work.state_vk.cu_deviceptr();
    let workspace = work.workspace.cu_deviceptr();
    let workspace_bytes = aot.workspace_bytes;
    let initial = if aot.cached {
        work.initial_vk.cu_deviceptr()
    } else {
        0
    };
    let launch = GdnAotLaunch {
        stream: core.stream.cu_stream() as u64,
        tokens,
        batch,
        key_heads,
        value_heads,
        q,
        k,
        v,
        alpha: provider_alpha,
        beta: provider_beta,
        output,
        cumulative: bindings.cumulative,
        state,
        workspace,
        workspace_bytes,
        initial,
    };
    *queued = true;
    // SAFETY: the preflight checked every live source/shape/pointer/capacity
    // and alignment fact. Each address belongs to an allocation in the owned
    // stream's context and stays resident through submission completion.
    unsafe {
        work.provider
            .forward(&launch)
            .map_err(|error| GdnEnqueueFailure {
                error,
                queued: *queued,
            })?;
    }

    if aot.write_final {
        let mut source = work.state_vk.cu_deviceptr();
        let mut destination = bindings.final_state;
        let mut batch = i64::from(plan.sequences);
        let mut value_heads = i64::from(plan.value_heads);
        let mut key_dim = GDN_KEY_DIM as i64;
        let mut value_dim = GDN_VALUE_DIM as i64;
        let mut state_args = [
            arg(&mut source),
            arg(&mut destination),
            arg(&mut batch),
            arg(&mut value_heads),
            arg(&mut key_dim),
            arg(&mut value_dim),
        ];
        launch_aot_state(
            &work.kernels.vk_to_kv,
            &core.stream,
            aot.state_grid,
            &mut state_args,
            queued,
        )?;
    }

    Ok(())
}

fn retain_inputs(
    core: &mut GdnCore,
    inputs: &GdnInputViews<'_>,
    state: &GdnState<'_>,
    slots: usize,
) -> Result<Vec<RawDeviceBuffer>, BackendError> {
    let mut retained = std::mem::take(&mut core.retained_storage);
    if retained.capacity() < slots {
        core.retained_storage = retained;
        return Err(start("input allocation retention slots are unavailable"));
    }
    retained.clear();
    retained.push(inputs.qkv.storage().clone());
    retained.push(inputs.z.storage().clone());
    retained.push(inputs.ab.storage().clone());
    if let GdnState::Cached(cached) = state {
        retained.push(cached.initial.storage().clone());
        retained.push(cached.history.rows.storage().clone());
    }
    Ok(retained)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GdnCachedMetadataLengths {
    cumulative: usize,
    sources: usize,
}

fn cached_metadata_lengths(cumulative: &[i32], sources: &[i64]) -> GdnCachedMetadataLengths {
    GdnCachedMetadataLengths {
        cumulative: cumulative.len(),
        sources: sources.len(),
    }
}

fn stage_host_metadata(
    core: &mut GdnCore,
    cumulative: &[u32],
    state: &GdnState<'_>,
) -> Result<(), BackendError> {
    stage_host_metadata_views(core, cumulative, state.borrowed())
}

fn stage_host_metadata_views(
    core: &mut GdnCore,
    cumulative: &[u32],
    state: GdnStateViews<'_>,
) -> Result<(), BackendError> {
    if cumulative.len() > core.host_cu.capacity() {
        return Err(infer(format!(
            "cumulative sequence metadata has {} entries, prepared capacity is {}",
            cumulative.len(),
            core.host_cu.capacity()
        )));
    }
    if let GdnStateViews::Cached {
        history_cumulative,
        sources,
        ..
    } = state
    {
        let lengths = cached_metadata_lengths(history_cumulative, sources);
        if lengths.cumulative > core.host_history_cu.capacity() {
            return Err(infer(format!(
                "cached history cumulative metadata has {} entries, prepared capacity is {}",
                lengths.cumulative,
                core.host_history_cu.capacity()
            )));
        }
        if lengths.sources > core.host_sources.capacity() {
            return Err(infer(format!(
                "cached source map has {} entries, prepared capacity is {}",
                lengths.sources,
                core.host_sources.capacity()
            )));
        }
    }
    core.metadata_generation = core
        .metadata_generation
        .checked_add(1)
        .ok_or_else(|| infer("GDN frame generation exhausted"))?;
    replace_host(
        &mut core.host_cu,
        cumulative,
        "cumulative sequence metadata",
    )?;
    if let GdnStateViews::Cached {
        history_cumulative,
        sources,
        ..
    } = state
    {
        replace_host(
            &mut core.host_history_cu,
            history_cumulative,
            "cached history cumulative metadata",
        )?;
        replace_host(&mut core.host_sources, sources, "cached source map")?;
    }
    Ok(())
}

// Completion and Drop may both visit this path. Return the reserved slot
// allocation at most once so the following submission stays allocation-free.
fn recycle_retained<R>(slots: &mut Vec<R>, retained: &mut Vec<R>) {
    if retained.capacity() != 0 {
        retained.clear();
        *slots = std::mem::take(retained);
    }
}

enum PendingDrain {
    Drained,
    Quarantined { error: BackendError },
}

fn settle_pending<C, R>(
    core: &mut Option<Box<C>>,
    retained: &mut Vec<R>,
    drain_result: Result<(), BackendError>,
) -> PendingDrain {
    match drain_result {
        Ok(()) => PendingDrain::Drained,
        Err(error) => {
            std::mem::forget(core.take());
            std::mem::forget(std::mem::take(retained));
            PendingDrain::Quarantined { error }
        }
    }
}

/// Borrowing owner of one queued submission.
pub struct GdnSubmission<'a> {
    prepared: &'a mut GdnPrepared,
    _inputs: GdnInputs<'a>,
    _state: GdnState<'a>,
    output_elements: usize,
    final_elements: Option<usize>,
    retained: Vec<RawDeviceBuffer>,
    completed: bool,
}

impl GdnSubmission<'_> {
    pub fn stream(&self) -> &Arc<Stream> {
        &self.prepared.stream
    }

    /// Borrow the actual fresh/cached BF16 gated output rows.
    pub fn output(&self) -> Result<DeviceBufferView<'_, bf16>, BackendError> {
        self.prepared
            .core_ref()?
            .output
            .slice(0..self.output_elements)
            .map_err(infer)
    }

    /// Borrow the optional sequence-indexed FP32 final state.
    pub fn final_state(&self) -> Result<Option<DeviceBufferView<'_, f32>>, BackendError> {
        match self.final_elements {
            Some(elements) => self
                .prepared
                .core_ref()?
                .final_state
                .slice(0..elements)
                .map(Some)
                .map_err(infer),
            None => Ok(None),
        }
    }

    pub fn query(&self) -> Result<bool, BackendError> {
        self.prepared.core_ref()?;
        if self.completed {
            return Ok(true);
        }
        self.prepared.stream.query().map_err(infer)
    }

    /// Explicit success-path completion and synchronization.
    pub fn complete(&mut self) -> Result<(), BackendError> {
        self.prepared.core_ref()?;
        if self.completed {
            return Ok(());
        }
        let drain_result = self.prepared.stream.synchronize().map_err(infer);
        match settle_pending(&mut self.prepared.core, &mut self.retained, drain_result) {
            PendingDrain::Drained => {
                self.completed = true;
                self.return_retained();
                Ok(())
            }
            PendingDrain::Quarantined { error } => {
                self.completed = true;
                Err(error)
            }
        }
    }

    fn return_retained(&mut self) {
        if let Some(core) = self.prepared.core.as_mut() {
            recycle_retained(&mut core.retained_storage, &mut self.retained);
        }
    }
}

impl Drop for GdnSubmission<'_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let drain_result = self.prepared.stream.synchronize().map_err(infer);
        if let PendingDrain::Drained =
            settle_pending(&mut self.prepared.core, &mut self.retained, drain_result)
        {
            self.completed = true;
            self.return_retained();
        }
    }
}

fn validate_weights(geometry: GdnGeometry, weights: &GdnWeights<'_>) -> Result<(), BackendError> {
    let conv_elements = product(geometry.channels(), geometry.kernel, "convolution weight")?;
    if weights.conv.len() != conv_elements {
        return Err(infer(format!(
            "convolution weight has {} elements, expected {conv_elements}",
            weights.conv.len()
        )));
    }
    if weights.a_log.len() != geometry.value_heads {
        return Err(infer(format!(
            "A_log has {} elements, expected {}",
            weights.a_log.len(),
            geometry.value_heads
        )));
    }
    if weights.dt_bias.len() != geometry.value_heads {
        return Err(infer(format!(
            "dt_bias has {} elements, expected {}",
            weights.dt_bias.len(),
            geometry.value_heads
        )));
    }
    if weights.norm.len() != GDN_VALUE_DIM {
        return Err(infer(format!(
            "norm weight has {} elements, expected {GDN_VALUE_DIM}",
            weights.norm.len()
        )));
    }
    for (index, &value) in weights.conv.iter().enumerate() {
        if !value.to_f32().is_finite() {
            return Err(infer(format!("convolution weight {index} is not finite")));
        }
    }
    for (index, &value) in weights.a_log.iter().enumerate() {
        if !value.is_finite() {
            return Err(infer(format!("A_log value {index} is not finite")));
        }
    }
    for (index, &value) in weights.dt_bias.iter().enumerate() {
        if !value.is_finite() {
            return Err(infer(format!("dt_bias value {index} is not finite")));
        }
    }
    for (index, &value) in weights.norm.iter().enumerate() {
        if !value.to_f32().is_finite() {
            return Err(infer(format!("norm weight {index} is not finite")));
        }
    }
    Ok(())
}

fn retained_storage_slots() -> Result<Vec<RawDeviceBuffer>, BackendError> {
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(5)
        .map_err(|_| start("input allocation retention slots are unavailable"))?;
    Ok(slots)
}

fn zeroed_host<T: Clone + Default>(len: usize, name: &str) -> Result<Vec<T>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| start(format!("{name} host metadata allocation failed")))?;
    values.resize(len, T::default());
    Ok(values)
}

fn replace_host<T: Copy>(
    destination: &mut Vec<T>,
    source: &[T],
    name: &str,
) -> Result<(), BackendError> {
    if source.len() > destination.capacity() {
        return Err(infer(format!(
            "{name} has {} entries, prepared capacity is {}",
            source.len(),
            destination.capacity()
        )));
    }
    destination.clear();
    destination.extend_from_slice(source);
    Ok(())
}

fn metadata_bytes<T>(len: usize) -> Result<usize, BackendError> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| infer("metadata upload byte extent overflows"))
}

fn upload_host_async<T: DeviceCopy>(
    stream: &Stream,
    destination: &DeviceBuffer<T>,
    source: &[T],
    bytes: usize,
    queued: &mut bool,
) -> Result<(), GdnEnqueueFailure> {
    if bytes == 0 {
        return Ok(());
    }
    *queued = true;
    // SAFETY: the host vector is owned by the prepared operator and remains
    // unchanged until this submission is completed or drained. The device
    // buffer is a prepared allocation with the same context as the stream.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            destination.cu_deviceptr(),
            source.as_ptr(),
            bytes,
            stream.cu_stream(),
        )
    }
    .map_err(|error| GdnEnqueueFailure {
        error: infer(error),
        queued: *queued,
    })
}

fn require_context(
    stream: &Arc<Stream>,
    context: &Arc<Context>,
    name: &str,
) -> Result<(), BackendError> {
    if !Arc::ptr_eq(stream.context(), context) {
        return Err(infer(format!("{name} belongs to a different CUDA context")));
    }
    Ok(())
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn launch_aot_state(
    function: &Function,
    stream: &Arc<Stream>,
    grid: (u32, u32, u32),
    args: &mut [*mut c_void],
    queued: &mut bool,
) -> Result<(), GdnEnqueueFailure> {
    if grid.0 == 0 || grid.1 != 1 || grid.2 != 1 {
        return Err(GdnEnqueueFailure {
            error: infer("invalid AOT state transpose grid"),
            queued: *queued,
        });
    }
    *queued = true;
    // SAFETY: the state tile ABI is a 32x32 block. Its shared transpose uses
    // both threadIdx dimensions; a flat 1024-thread block is not equivalent.
    unsafe {
        cuda_core::launch_kernel(
            function.cu_function(),
            grid,
            AOT_STATE_BLOCK,
            0,
            stream.cu_stream(),
            args,
        )
    }
    .map_err(|error| GdnEnqueueFailure {
        error: infer(error),
        queued: *queued,
    })
}

fn launch_raw(
    function: &Function,
    stream: &Arc<Stream>,
    grid: (u32, u32, u32),
    threads: u32,
    args: &mut [*mut c_void],
    queued: &mut bool,
) -> Result<(), GdnEnqueueFailure> {
    if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
        return Err(GdnEnqueueFailure {
            error: infer("refusing a zero-extent native grid"),
            queued: *queued,
        });
    }
    *queued = true;
    // SAFETY: every pointer argument was derived from a checked allocation in
    // the prepared stream's context, and the argument list matches the exact
    // raw kernel signature for this function.
    unsafe {
        cuda_core::launch_kernel(
            function.cu_function(),
            grid,
            (threads, 1, 1),
            0,
            stream.cu_stream(),
            args,
        )
    }
    .map_err(|error| GdnEnqueueFailure {
        error: infer(error),
        queued: *queued,
    })
}

#[derive(Deserialize)]
struct GdnBundleManifest {
    abi_version: u32,
    operator: String,
    compile_only: bool,
    device_calls: bool,
    architectures: Vec<String>,
    exports: Vec<String>,
    sources: BTreeMap<String, String>,
    toolkit: serde_json::Value,
    flags: Vec<String>,
    torch_symbols: Vec<String>,
    compilations: Vec<GdnCompilationRecord>,
}

#[derive(Deserialize)]
struct GdnCompilationRecord {
    architecture: String,
    cubin: String,
    cubin_sha256: String,
    cubin_bytes: u64,
    exports: Vec<String>,
    #[serde(default)]
    torch_symbols: Vec<String>,
}

impl GdnBundleManifest {
    fn parse(text: &str) -> Result<Self, BackendError> {
        let manifest: Self = serde_json::from_str(text)
            .map_err(|error| start(format!("invalid bundle manifest: {error}")))?;
        if manifest.abi_version != 1 {
            return Err(start(format!(
                "unsupported bundle ABI {}, expected 1",
                manifest.abi_version
            )));
        }
        if manifest.operator != "qwen35_gdn_native" {
            return Err(start(format!(
                "bundle operator {:?} is not qwen35_gdn_native",
                manifest.operator
            )));
        }
        if !manifest.compile_only || manifest.device_calls {
            return Err(start("bundle must be compile-only with no device calls"));
        }
        if manifest.architectures != REQUIRED_ARCHITECTURES {
            return Err(start(
                "bundle must contain exactly sm_90, sm_100, and sm_120",
            ));
        }
        let actual: BTreeSet<&str> = manifest.exports.iter().map(String::as_str).collect();
        let expected: BTreeSet<&str> = REQUIRED_EXPORTS.into_iter().collect();
        if actual != expected || actual.len() != manifest.exports.len() {
            return Err(start(format!(
                "bundle exports {:?} do not match the required set",
                manifest.exports
            )));
        }
        if !manifest.torch_symbols.is_empty() {
            return Err(start("bundle records linked Torch symbols"));
        }
        let source_names: BTreeSet<&str> = manifest.sources.keys().map(String::as_str).collect();
        let required_sources: BTreeSet<&str> = REQUIRED_SOURCES.into_iter().collect();
        if source_names != required_sources {
            return Err(start(format!(
                "bundle source records {:?} do not match the required set",
                manifest.sources.keys().collect::<Vec<_>>()
            )));
        }
        if !manifest.sources.values().all(|value| is_sha256(value)) {
            return Err(start("bundle source hashes are missing or malformed"));
        }
        if !manifest.flags.iter().map(String::as_str).eq(REQUIRED_FLAGS) {
            return Err(start(
                "bundle arithmetic flags differ from the exact native producer contract",
            ));
        }
        let version = manifest
            .toolkit
            .get("version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if version.trim().is_empty() {
            return Err(start("bundle toolkit version is not recorded"));
        }
        if manifest.compilations.len() != REQUIRED_ARCHITECTURES.len() {
            return Err(start(format!(
                "bundle has {} compilation records, expected {}",
                manifest.compilations.len(),
                REQUIRED_ARCHITECTURES.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for (index, record) in manifest.compilations.iter().enumerate() {
            let expected_architecture = REQUIRED_ARCHITECTURES[index];
            if record.architecture != expected_architecture {
                return Err(start(format!(
                    "compilation record {index} is {:?}, expected {expected_architecture}",
                    record.architecture
                )));
            }
            if !seen.insert(record.architecture.as_str()) {
                return Err(start(format!(
                    "duplicate compilation record for {}",
                    record.architecture
                )));
            }
            let expected_path = format!("{expected_architecture}/kernels.cubin");
            if record.cubin != expected_path || !is_safe_cubin_path(&record.cubin) {
                return Err(start(format!(
                    "compilation record {index} has unsafe cubin path {:?}",
                    record.cubin
                )));
            }
            if !is_sha256(&record.cubin_sha256) {
                return Err(start(format!(
                    "compilation record {index} has a malformed cubin hash"
                )));
            }
            if record.cubin_bytes == 0 {
                return Err(start(format!(
                    "compilation record {index} has zero cubin bytes"
                )));
            }
            let record_exports: BTreeSet<&str> =
                record.exports.iter().map(String::as_str).collect();
            if record_exports != expected || record_exports.len() != record.exports.len() {
                return Err(start(format!(
                    "compilation record {index} exports {:?} do not match the required set",
                    record.exports
                )));
            }
            if !record.torch_symbols.is_empty() {
                return Err(start(format!(
                    "compilation record {index} records linked Torch symbols"
                )));
            }
        }
        Ok(manifest)
    }

    fn validate_sources(&self, bundle: &Path) -> Result<(), BackendError> {
        for (name, expected) in &self.sources {
            if name == "qwen35_gdn_native.cu" {
                let actual = sha256_hex(GDN_NATIVE_SOURCE);
                if !expected.eq_ignore_ascii_case(&actual) {
                    return Err(start(
                        "bundle native source hash does not match the retained source",
                    ));
                }
                continue;
            }
            let source_path = bundle.join(name);
            let bytes = std::fs::read(&source_path)
                .map_err(|error| start(format!("{}: {error}", source_path.display())))?;
            let actual = sha256_hex(&bytes);
            if !expected.eq_ignore_ascii_case(&actual) {
                return Err(start(format!(
                    "bundle source {} hash does not match its contents",
                    source_path.display()
                )));
            }
        }
        Ok(())
    }
}

struct GdnValidatedBundle {
    cubins: BTreeMap<String, Vec<u8>>,
}

impl GdnValidatedBundle {
    fn load(bundle: &Path) -> Result<Self, BackendError> {
        let manifest_path = bundle.join("manifest.json");
        let manifest_bytes = std::fs::read(&manifest_path)
            .map_err(|error| start(format!("{}: {error}", manifest_path.display())))?;
        let manifest_text = std::str::from_utf8(&manifest_bytes)
            .map_err(|error| start(format!("{} is not UTF-8: {error}", manifest_path.display())))?;
        let manifest = GdnBundleManifest::parse(manifest_text)?;
        manifest.validate_sources(bundle)?;
        let canonical_bundle = bundle
            .canonicalize()
            .map_err(|error| start(format!("{}: {error}", bundle.display())))?;

        let mut cubins = BTreeMap::new();
        for record in &manifest.compilations {
            let cubin_path = bundle.join(&record.cubin);
            let canonical_cubin = cubin_path
                .canonicalize()
                .map_err(|error| start(format!("{}: {error}", cubin_path.display())))?;
            if !canonical_cubin.starts_with(&canonical_bundle) {
                return Err(start(format!(
                    "cubin path {} escapes the bundle directory",
                    cubin_path.display()
                )));
            }
            let image = std::fs::read(&cubin_path)
                .map_err(|error| start(format!("{}: {error}", cubin_path.display())))?;
            let actual_bytes =
                u64::try_from(image.len()).map_err(|_| start("cubin byte extent exceeds u64"))?;
            if actual_bytes != record.cubin_bytes {
                return Err(start(format!(
                    "{} has {actual_bytes} bytes, manifest records {}",
                    cubin_path.display(),
                    record.cubin_bytes
                )));
            }
            let actual_hash = sha256_hex(&image);
            if !record.cubin_sha256.eq_ignore_ascii_case(&actual_hash) {
                return Err(start(format!(
                    "{} does not match its manifest cubin hash",
                    cubin_path.display()
                )));
            }
            cubins.insert(record.architecture.clone(), image);
        }
        Ok(Self { cubins })
    }

    fn image(&self, architecture: &str) -> Option<&[u8]> {
        self.cubins.get(architecture).map(Vec::as_slice)
    }
}

#[derive(Deserialize)]
struct GdnAotBundleManifest {
    abi_version: u32,
    operator: String,
    compile_only: bool,
    device_calls: bool,
    architectures: Vec<String>,
    exports: Vec<String>,
    sources: BTreeMap<String, String>,
    toolkit: serde_json::Value,
    flags: Vec<String>,
    torch_symbols: Vec<String>,
    compilations: Vec<GdnCompilationRecord>,
}

impl GdnAotBundleManifest {
    fn parse(text: &str) -> Result<Self, BackendError> {
        let manifest: Self = serde_json::from_str(text)
            .map_err(|error| start(format!("invalid AOT bundle manifest: {error}")))?;
        if manifest.abi_version != 1 {
            return Err(start(format!(
                "unsupported AOT bundle ABI {}, expected 1",
                manifest.abi_version
            )));
        }
        if manifest.operator != AOT_OPERATOR {
            return Err(start(format!(
                "AOT bundle operator {:?} is not {AOT_OPERATOR:?}",
                manifest.operator
            )));
        }
        if !manifest.compile_only || manifest.device_calls {
            return Err(start(
                "AOT bundle must be compile-only with no device calls",
            ));
        }
        if manifest.architectures != REQUIRED_ARCHITECTURES {
            return Err(start(
                "AOT bundle must contain exactly sm_90, sm_100, and sm_120",
            ));
        }
        let actual: BTreeSet<&str> = manifest.exports.iter().map(String::as_str).collect();
        let expected: BTreeSet<&str> = AOT_REQUIRED_EXPORTS.into_iter().collect();
        if actual != expected || actual.len() != manifest.exports.len() {
            return Err(start(format!(
                "AOT bundle exports {:?} do not match the required set",
                manifest.exports
            )));
        }
        if !manifest.torch_symbols.is_empty() {
            return Err(start("AOT bundle records linked Torch symbols"));
        }
        let source_names: BTreeSet<&str> = manifest.sources.keys().map(String::as_str).collect();
        let required_sources: BTreeSet<&str> = AOT_REQUIRED_SOURCES.into_iter().collect();
        if source_names != required_sources {
            return Err(start(format!(
                "AOT bundle source records {:?} do not match the required set",
                manifest.sources.keys().collect::<Vec<_>>()
            )));
        }
        if !manifest.sources.values().all(|value| is_sha256(value)) {
            return Err(start("AOT bundle source hashes are missing or malformed"));
        }
        if !manifest.flags.iter().map(String::as_str).eq(REQUIRED_FLAGS) {
            return Err(start(
                "AOT bundle arithmetic flags differ from the exact producer contract",
            ));
        }
        let version = manifest
            .toolkit
            .get("version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if version.trim().is_empty() {
            return Err(start("AOT bundle toolkit version is not recorded"));
        }
        if manifest.compilations.len() != REQUIRED_ARCHITECTURES.len() {
            return Err(start(format!(
                "AOT bundle has {} compilation records, expected {}",
                manifest.compilations.len(),
                REQUIRED_ARCHITECTURES.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for (index, record) in manifest.compilations.iter().enumerate() {
            let expected_architecture = REQUIRED_ARCHITECTURES[index];
            if record.architecture != expected_architecture {
                return Err(start(format!(
                    "AOT compilation record {index} is {:?}, expected {expected_architecture}",
                    record.architecture
                )));
            }
            if !seen.insert(record.architecture.as_str()) {
                return Err(start(format!(
                    "duplicate AOT compilation record for {}",
                    record.architecture
                )));
            }
            let expected_path = format!("{expected_architecture}/kernels.cubin");
            if record.cubin != expected_path || !is_safe_cubin_path(&record.cubin) {
                return Err(start(format!(
                    "AOT compilation record {index} has unsafe cubin path {:?}",
                    record.cubin
                )));
            }
            if !is_sha256(&record.cubin_sha256) {
                return Err(start(format!(
                    "AOT compilation record {index} has a malformed cubin hash"
                )));
            }
            if record.cubin_bytes == 0 {
                return Err(start(format!(
                    "AOT compilation record {index} has zero cubin bytes"
                )));
            }
            let record_exports: BTreeSet<&str> =
                record.exports.iter().map(String::as_str).collect();
            if record_exports != expected || record_exports.len() != record.exports.len() {
                return Err(start(format!(
                    "AOT compilation record {index} exports {:?} do not match the required set",
                    record.exports
                )));
            }
            if !record.torch_symbols.is_empty() {
                return Err(start(format!(
                    "AOT compilation record {index} records linked Torch symbols"
                )));
            }
        }
        Ok(manifest)
    }

    fn validate_sources(&self, bundle: &Path) -> Result<(), BackendError> {
        for (name, expected) in &self.sources {
            if name == "qwen35_gdn_aot.cu" {
                let actual = sha256_hex(GDN_AOT_SOURCE);
                if !expected.eq_ignore_ascii_case(&actual) {
                    return Err(start(
                        "AOT bundle source hash does not match the retained source",
                    ));
                }
                continue;
            }
            let source_path = bundle.join(name);
            let bytes = std::fs::read(&source_path)
                .map_err(|error| start(format!("{}: {error}", source_path.display())))?;
            let actual = sha256_hex(&bytes);
            if !expected.eq_ignore_ascii_case(&actual) {
                return Err(start(format!(
                    "AOT bundle source {} hash does not match its contents",
                    source_path.display()
                )));
            }
        }
        Ok(())
    }
}

struct GdnAotValidatedBundle {
    cubins: BTreeMap<String, Vec<u8>>,
}

impl GdnAotValidatedBundle {
    fn load(bundle: &Path) -> Result<Self, BackendError> {
        let manifest_path = bundle.join("manifest.json");
        let manifest_bytes = std::fs::read(&manifest_path)
            .map_err(|error| start(format!("{}: {error}", manifest_path.display())))?;
        let manifest_text = std::str::from_utf8(&manifest_bytes)
            .map_err(|error| start(format!("{} is not UTF-8: {error}", manifest_path.display())))?;
        let manifest = GdnAotBundleManifest::parse(manifest_text)?;
        manifest.validate_sources(bundle)?;
        let canonical_bundle = bundle
            .canonicalize()
            .map_err(|error| start(format!("{}: {error}", bundle.display())))?;

        let mut cubins = BTreeMap::new();
        for record in &manifest.compilations {
            let cubin_path = bundle.join(&record.cubin);
            let canonical_cubin = cubin_path
                .canonicalize()
                .map_err(|error| start(format!("{}: {error}", cubin_path.display())))?;
            if !canonical_cubin.starts_with(&canonical_bundle) {
                return Err(start(format!(
                    "AOT cubin path {} escapes the bundle directory",
                    cubin_path.display()
                )));
            }
            let image = std::fs::read(&cubin_path)
                .map_err(|error| start(format!("{}: {error}", cubin_path.display())))?;
            let actual_bytes = u64::try_from(image.len())
                .map_err(|_| start("AOT cubin byte extent exceeds u64"))?;
            if actual_bytes != record.cubin_bytes {
                return Err(start(format!(
                    "{} has {actual_bytes} bytes, manifest records {}",
                    cubin_path.display(),
                    record.cubin_bytes
                )));
            }
            let actual_hash = sha256_hex(&image);
            if !record.cubin_sha256.eq_ignore_ascii_case(&actual_hash) {
                return Err(start(format!(
                    "{} does not match its manifest cubin hash",
                    cubin_path.display()
                )));
            }
            cubins.insert(record.architecture.clone(), image);
        }
        Ok(Self { cubins })
    }

    fn image(&self, architecture: &str) -> Option<&[u8]> {
        self.cubins.get(architecture).map(Vec::as_slice)
    }
}

/// Validate the separate three-export packing bundle entirely on the CPU,
/// including every architecture, export, flag, source hash, cubin hash and
/// byte count, and safe relative path, before any module is loaded.
/// Check a provider descriptor and all declared library bytes without loading
/// a library or querying CUDA. Runtime construction samples actual device facts.
pub fn validate_aot_provider_bundle(
    bundle: &Path,
    major: i32,
    sm_count: i32,
) -> Result<(), BackendError> {
    GdnAotPlan::load(
        bundle,
        DeviceFacts {
            ordinal: 0,
            major,
            sm_count,
        },
    )
    .map(|_| ())
}

pub fn validate_aot_bundle(bundle: &Path) -> Result<(), BackendError> {
    GdnAotValidatedBundle::load(bundle).map(|_| ())
}

fn is_safe_cubin_path(value: &str) -> bool {
    let mut components = Path::new(value).components();
    matches!(
        (components.next(), components.next(), components.next()),
        (
            Some(std::path::Component::Normal(_)),
            Some(std::path::Component::Normal(name)),
            None
        ) if name == "kernels.cubin"
    )
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> GdnGeometry {
        GdnGeometry::new(2, 4, 4).expect("bounded geometry")
    }

    fn capacity(tokens: usize, sequences: usize, history: usize) -> GdnCapacity {
        GdnCapacity::new(tokens, sequences, history).expect("bounded capacity")
    }

    fn extent(elements: usize, row_stride: usize, byte_alignment: usize) -> GdnTensorExtent {
        GdnTensorExtent {
            elements,
            row_stride,
            byte_alignment,
        }
    }

    #[test]
    fn empty_aot_building_finishes_without_a_provider_owner() {
        assert!(GdnAotBuilding::new().finish().unwrap().is_none());
    }

    fn compilation(architecture: &str) -> serde_json::Value {
        serde_json::json!({
            "architecture": architecture,
            "cubin": format!("{architecture}/kernels.cubin"),
            "cubin_sha256": "d".repeat(64),
            "cubin_bytes": 128,
            "exports": REQUIRED_EXPORTS,
            "torch_symbols": [],
        })
    }

    fn compilation_records() -> serde_json::Value {
        serde_json::json!(REQUIRED_ARCHITECTURES
            .iter()
            .map(|architecture| compilation(architecture))
            .collect::<Vec<_>>())
    }

    #[test]
    fn borrowed_phase_ranges_preserve_offsets_and_reject_wrapping_addresses() {
        assert_eq!(
            borrowed_range(0x1000, 16384, 2048..8192, 2).unwrap(),
            (0x2000, 6144)
        );
        assert_eq!(
            borrowed_range(0x1000, 16384, 8192..8288, 2).unwrap(),
            (0x5000, 96)
        );
        assert!(borrowed_range(0x1000, 100, 99..101, 2).is_err());
        let reversed = std::ops::Range { start: 8, end: 7 };
        assert!(borrowed_range(0x1000, 100, reversed, 2).is_err());
        assert!(borrowed_range(u64::MAX - 1, 100, 1..2, 2).is_err());
        assert!(borrowed_range(0x1000, usize::MAX, usize::MAX / 2 + 1..usize::MAX, 2).is_err());
    }

    #[test]
    fn rejects_unsupported_head_and_kernel_shapes() {
        for (kh, vh, kernel) in [
            (0, 1, 4),
            (2, 1, 4),
            (2, 129, 4),
            (2, 5, 4),
            (2, 4, 0),
            (2, 4, 17),
        ] {
            assert!(
                GdnGeometry::new(kh, vh, kernel).is_err(),
                "kh={kh} vh={vh} kernel={kernel} must be rejected"
            );
        }
        let g = geometry();
        assert_eq!(g.channels(), (2 * 2 + 4) * 128);
        assert_eq!(g.cached_lanes(63).expect("lane choice"), 16);
        assert_eq!(g.cached_lanes(64).expect("lane choice"), 8);
    }

    #[test]
    fn scratch_layout_enforces_native_extent_bounds() {
        let g = GdnGeometry::new(1, 2, 3).expect("geometry");
        let too_many_tokens = GdnCapacity::new(GDN_MAX_TOKENS, 1, 0).expect("capacity");
        assert!(GdnScratchLayout::new(g, too_many_tokens).is_err());
        assert!(GdnCapacity::new(i32::MAX as usize + 1, 1, 0).is_err());
        assert!(GdnCapacity::new(8, 9, 0).is_err());

        let ok = GdnScratchLayout::new(g, capacity(6, 2, 3)).expect("layout");
        assert_eq!(ok.convolution_elements(), 6 * g.channels());
        assert_eq!(ok.qk_elements(), 6 * 256);
        assert_eq!(ok.decay_beta_elements(), 6 * 2 * 2);
        assert_eq!(ok.recurrence_elements(), 6 * 2 * 128);
        assert_eq!(ok.output_elements(), 6 * 2 * 128);
        assert_eq!(ok.final_state_elements(), 2 * 2 * 128 * 128);
        assert_eq!(ok.cumulative_elements(), 3);
        assert_eq!(ok.history_cumulative_elements(), 4);
    }

    #[test]
    fn ragged_layout_and_request_plan_use_native_token_and_state_extents() {
        let g = GdnGeometry::new(2, 4, 3).expect("geometry");
        let c = capacity(72, 2, 2);
        let layout = GdnScratchLayout::new(g, c).expect("layout");
        assert_eq!(layout.recurrence_elements(), 72 * 4 * 128);
        assert_eq!(layout.output_elements(), 72 * 4 * 128);
        assert_eq!(layout.final_state_elements(), 2 * 4 * 128 * 128);
        assert_eq!(layout.recurrence_elements(), 36_864);
        assert_eq!(layout.output_elements(), 36_864);
        assert_eq!(layout.final_state_elements(), 131_072);

        let plan = GdnPlan::new(g, c, &[0, 31, 72]).expect("ragged plan");
        let extents = request_extents(g, plan, true).expect("request extents");
        assert_eq!(extents.output_elements, 72 * 4 * 128);
        assert_eq!(extents.output_elements, 36_864);
        assert_eq!(extents.final_elements, Some(2 * 4 * 128 * 128));
        assert_eq!(extents.final_elements, Some(131_072));

        let no_final = request_extents(g, plan, false).expect("request extents");
        assert_eq!(no_final.final_elements, None);
        assert_eq!(no_final.output_elements, 36_864);
    }

    #[test]
    fn cached_history_element_extent_maps_to_row_count() {
        let channels = geometry().channels();
        assert_eq!(history_row_count(3 * channels, channels).expect("rows"), 3);
        assert_eq!(history_row_count(0, channels).expect("empty rows"), 0);
        assert!(history_row_count(3 * channels + 1, channels).is_err());
        assert!(history_row_count(1, 0).is_err());
    }

    #[test]
    fn borrowed_fresh_state_and_cached_metadata_lengths_are_view_independent() {
        let fresh = GdnState::Fresh;
        let views = fresh.borrowed();
        assert!(matches!(views, GdnStateViews::Fresh));
        let copied = views;
        assert!(matches!(copied, GdnStateViews::Fresh));

        let cumulative = [0i32, 2, 3];
        let sources = [1i64, 1];
        let lengths = cached_metadata_lengths(&cumulative, &sources);
        assert_eq!(
            lengths,
            GdnCachedMetadataLengths {
                cumulative: 3,
                sources: 2
            }
        );
        let plan =
            GdnHistoryPlan::new(geometry(), 2, &sources, &cumulative, 3).expect("history plan");
        assert_eq!(plan.initial_sequences(), lengths.cumulative - 1);
        assert_eq!(plan.history_rows(), 3);
        assert_eq!(
            plan.history_rows(),
            usize::try_from(cumulative[cumulative.len() - 1]).expect("row count")
        );
        assert_eq!(
            cached_metadata_lengths(&[], &[]),
            GdnCachedMetadataLengths {
                cumulative: 0,
                sources: 0
            }
        );
    }

    #[test]
    fn rejects_ragged_qkv_and_noncontiguous_z_layouts() {
        let g = geometry();
        let channels = g.channels();
        let tokens = 5;
        let qkv_stride = channels + 4;
        let qkv = extent((tokens - 1) * qkv_stride + channels, qkv_stride, 8);
        let z_width = g.value_heads * GDN_VALUE_DIM;
        let z = extent(tokens * z_width, z_width, 2);
        let ab_width = g.value_heads * 2;
        let ab = extent(tokens * ab_width, ab_width, 2);
        assert!(GdnInputLayout::admit(g, tokens, qkv, z, ab).is_ok());

        let mut bad_qkv = qkv;
        bad_qkv.elements -= 1;
        assert!(GdnInputLayout::admit(g, tokens, bad_qkv, z, ab).is_err());
        let mut bad_qkv = qkv;
        bad_qkv.byte_alignment = 4;
        assert!(GdnInputLayout::admit(g, tokens, bad_qkv, z, ab).is_err());
        let mut bad_qkv = qkv;
        bad_qkv.row_stride += 1;
        assert!(GdnInputLayout::admit(g, tokens, bad_qkv, z, ab).is_err());

        let padded_z = extent(tokens * (z_width + 4), z_width + 4, 2);
        assert!(GdnInputLayout::admit(g, tokens, qkv, padded_z, ab).is_err());
        let short_ab = extent(tokens * ab_width - 1, ab_width, 2);
        assert!(GdnInputLayout::admit(g, tokens, qkv, z, short_ab).is_err());
    }

    #[test]
    fn sequence_metadata_requires_strict_origin_and_token_bound() {
        let g = geometry();
        let c = capacity(8, 3, 2);
        assert!(GdnPlan::new(g, c, &[0, 3, 7, 8]).is_ok());
        for cumulative in [&[1u32, 2, 3][..], &[0, 0, 3][..], &[0, 4, 4][..], &[0][..]] {
            assert!(GdnPlan::new(g, c, cumulative).is_err());
        }
        assert!(GdnPlan::new(g, capacity(7, 3, 2), &[0, 3, 7, 8]).is_err());
        assert!(GdnPlan::new(g, capacity(8, 2, 2), &[0, 3, 7, 8]).is_err());
        let plan = GdnPlan::new(g, c, &[0, 3, 7]).expect("plan");
        assert_eq!(plan.tokens(), 7);
        assert_eq!(plan.sequences(), 2);
        assert_eq!(plan.max_sequence(), 4);
    }

    #[test]
    fn cached_history_maps_sources_and_final_rows_by_sequence() {
        let g = geometry();
        let sources = [1i64, 0i64];
        let history_cumulative = [0i32, 2, 3];
        let plan = GdnHistoryPlan::new(g, 2, &sources, &history_cumulative, 3)
            .expect("cached history plan");
        assert_eq!(plan.initial_sequences(), 2);
        assert_eq!(plan.history_rows(), 3);
        assert_eq!(
            plan.initial_elements(),
            2 * g.value_heads * g.state_row_width()
        );
        assert_eq!(plan.history_elements(), 3 * g.channels());
        assert_eq!(
            plan.final_elements(),
            2 * g.value_heads * g.state_row_width()
        );

        assert!(GdnHistoryPlan::new(g, 2, &[2, 0], &history_cumulative, 3).is_err());
        assert!(GdnHistoryPlan::new(g, 2, &[0], &history_cumulative, 3).is_err());
        assert!(GdnHistoryPlan::new(g, 2, &sources, &[0, 2, 1], 2).is_err());
        assert!(GdnHistoryPlan::new(g, 2, &sources, &[0, 4, 5], 5).is_err());
        assert!(GdnHistoryPlan::new(g, 2, &sources, &[0, 2, 3], 4).is_err());
        assert!(GdnHistoryPlan::new(g, 2, &sources, &[0, 5, 6], 3).is_err());
        assert!(GdnHistoryPlan::new(g, 2, &sources, &[0, 2, 2], 2).is_ok());
        assert!(GdnHistoryPlan::new(g, 2, &sources, &[0, 2, 3, 3], 3).is_ok());

        let first = final_state_offset(g, 0).expect("sequence zero");
        let second = final_state_offset(g, 1).expect("sequence one");
        assert_eq!(first, 0);
        assert_eq!(second, g.value_heads * g.state_row_width());
        assert_ne!(
            second,
            final_state_offset(g, plan.initial_sequences()).expect("source row")
        );
    }

    #[test]
    fn weight_admission_rejects_wrong_extents_and_nonfinite_values() {
        let g = geometry();
        let conv = vec![bf16::from_f32(0.25); g.channels() * g.kernel];
        let a_log = vec![0.5f32; g.value_heads];
        let dt_bias = vec![0.125f32; g.value_heads];
        let norm = vec![bf16::from_f32(1.0); GDN_VALUE_DIM];
        let weights = GdnWeights {
            conv: &conv,
            a_log: &a_log,
            dt_bias: &dt_bias,
            norm: &norm,
        };
        assert!(validate_weights(g, &weights).is_ok());

        let short_conv = &conv[..conv.len() - 1];
        assert!(validate_weights(
            g,
            &GdnWeights {
                conv: short_conv,
                ..GdnWeights {
                    conv: &conv,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    norm: &norm,
                }
            }
        )
        .is_err());

        let mut nonfinite_a = a_log.clone();
        nonfinite_a[0] = f32::NAN;
        assert!(validate_weights(
            g,
            &GdnWeights {
                conv: &conv,
                a_log: &nonfinite_a,
                dt_bias: &dt_bias,
                norm: &norm,
            }
        )
        .is_err());

        let mut nonfinite_conv = conv.clone();
        nonfinite_conv[0] = bf16::INFINITY;
        assert!(validate_weights(
            g,
            &GdnWeights {
                conv: &nonfinite_conv,
                a_log: &a_log,
                dt_bias: &dt_bias,
                norm: &norm,
            }
        )
        .is_err());
    }

    #[test]
    fn rejects_invalid_epsilon_and_wrong_bundle_manifests() {
        assert!(validate_epsilon(f32::NAN).is_err());
        assert!(validate_epsilon(f32::INFINITY).is_err());
        assert!(validate_epsilon(0.0).is_err());
        assert!(validate_epsilon(-1.0).is_err());
        assert!(validate_epsilon(1e-6).is_ok());

        let valid = serde_json::json!({
            "abi_version": 1,
            "operator": "qwen35_gdn_native",
            "compile_only": true,
            "device_calls": false,
            "architectures": ["sm_90", "sm_100", "sm_120"],
            "exports": REQUIRED_EXPORTS,
            "sources": {
                "qwen35_gdn_native.cu": "a".repeat(64),
                "NOTICE": "b".repeat(64),
                "LICENSE.candle-MIT": "c".repeat(64),
            },
            "toolkit": {"nvcc": "/usr/local/cuda/bin/nvcc", "version": "release 13.1"},
            "flags": REQUIRED_FLAGS,
            "torch_symbols": [],
            "compilations": compilation_records(),
        });
        assert!(
            GdnBundleManifest::parse(&valid.to_string()).is_ok(),
            "valid manifest must parse"
        );

        let mut missing = valid.clone();
        missing["exports"] = serde_json::json!(&REQUIRED_EXPORTS[..7]);
        assert!(GdnBundleManifest::parse(&missing.to_string()).is_err());

        let mut extra_arch = valid.clone();
        extra_arch["architectures"] = serde_json::json!(["sm_90", "sm_100", "sm_120", "sm_80"]);
        assert!(GdnBundleManifest::parse(&extra_arch.to_string()).is_err());

        let mut torch = valid.clone();
        torch["torch_symbols"] = serde_json::json!(["at::Tensor"]);
        assert!(GdnBundleManifest::parse(&torch.to_string()).is_err());

        let mut duplicate = valid.clone();
        let first = compilation("sm_90");
        duplicate["compilations"] =
            serde_json::json!([first.clone(), first, compilation("sm_120"),]);
        assert!(GdnBundleManifest::parse(&duplicate.to_string()).is_err());

        let mut escape = valid.clone();
        escape["compilations"][0]["cubin"] = serde_json::json!("../sm_90/kernels.cubin");
        assert!(GdnBundleManifest::parse(&escape.to_string()).is_err());

        let mut fast = valid;
        fast["flags"] = serde_json::json!(["-O3", "--cubin", "--use_fast_math"]);
        assert!(GdnBundleManifest::parse(&fast.to_string()).is_err());
    }

    #[test]
    fn failed_drain_quarantines_the_whole_owner_and_retained_inputs() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Tracked {
            drops: Arc<AtomicUsize>,
        }

        impl Drop for Tracked {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let mut core = Some(Box::new(Tracked {
            drops: drops.clone(),
        }));
        let mut retained = vec![
            Tracked {
                drops: drops.clone(),
            },
            Tracked {
                drops: drops.clone(),
            },
        ];
        let outcome = settle_pending(
            &mut core,
            &mut retained,
            Err(infer("injected drain failure")),
        );
        assert!(matches!(outcome, PendingDrain::Quarantined { .. }));
        assert!(core.is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        let mut core = Some(Box::new(Tracked {
            drops: drops.clone(),
        }));
        let mut retained = vec![Tracked {
            drops: drops.clone(),
        }];
        let outcome = settle_pending(&mut core, &mut retained, Ok(()));
        assert!(matches!(outcome, PendingDrain::Drained));
        assert!(core.is_some());
        drop(core.take());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(retained);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    const V1_BUNDLE: &str = "/tmp/infinity-oxide-parity-20261101/qwen35-gdn-foundation-v1/kernels";

    fn copy_v1_bundle(destination: &std::path::Path) {
        let source = std::path::Path::new(V1_BUNDLE);
        for relative in [
            "manifest.json",
            "NOTICE",
            "LICENSE.candle-MIT",
            "sm_90/kernels.cubin",
            "sm_100/kernels.cubin",
            "sm_120/kernels.cubin",
        ] {
            let target = destination.join(relative);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).expect("create bundle subdirectory");
            }
            std::fs::copy(source.join(relative), &target).expect("copy V1 bundle file");
        }
    }

    fn write_manifest(destination: &std::path::Path, manifest: &serde_json::Value) {
        std::fs::write(
            destination.join("manifest.json"),
            serde_json::to_vec_pretty(manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
    }

    #[test]
    fn completed_submission_drop_preserves_retention_capacity_for_next_request() {
        let mut slots = Vec::with_capacity(5);
        let allocation = slots.as_ptr();
        let mut first = std::mem::take(&mut slots);
        first.extend([1u32, 2, 3, 4, 5]);
        recycle_retained(&mut slots, &mut first);
        assert_eq!(slots.as_ptr(), allocation);
        assert!(slots.capacity() >= 5);
        assert!(slots.is_empty());
        // Explicit complete followed by Drop must not replace the returned
        // slot allocation with the submission's now-empty vector.
        recycle_retained(&mut slots, &mut first);
        assert_eq!(slots.as_ptr(), allocation);
        assert!(slots.capacity() >= 5);
        let mut second = std::mem::take(&mut slots);
        second.extend([6u32, 7, 8]);
        recycle_retained(&mut slots, &mut second);
        assert_eq!(slots.as_ptr(), allocation);
        assert!(slots.capacity() >= 5);
    }

    #[test]
    fn actual_v1_bundle_validates_and_rejects_mutated_metadata() {
        let source = std::path::Path::new(V1_BUNDLE);
        if !source.join("manifest.json").is_file() {
            eprintln!("V1 generated GDN bundle is unavailable; skipping CPU fixture");
            return;
        }
        let destination =
            std::env::temp_dir().join(format!("gdn-v2-bundle-{}-{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&destination);
        std::fs::create_dir_all(&destination).expect("create temporary bundle");
        copy_v1_bundle(&destination);

        assert!(
            GdnKernelSet::validate_bundle(&destination).is_ok(),
            "actual V1 manifest and cubins must validate"
        );

        let original: serde_json::Value = serde_json::from_slice(
            &std::fs::read(source.join("manifest.json")).expect("read V1 manifest"),
        )
        .expect("parse V1 manifest");

        let mut checksum = original.clone();
        checksum["compilations"][0]["cubin_sha256"] = serde_json::json!("0".repeat(64));
        write_manifest(&destination, &checksum);
        assert!(GdnKernelSet::validate_bundle(&destination).is_err());

        let mut size = original.clone();
        size["compilations"][1]["cubin_bytes"] = serde_json::json!(1);
        write_manifest(&destination, &size);
        assert!(GdnKernelSet::validate_bundle(&destination).is_err());

        let mut duplicate = original.clone();
        let first = duplicate["compilations"][0].clone();
        duplicate["compilations"]
            .as_array_mut()
            .unwrap()
            .push(first);
        write_manifest(&destination, &duplicate);
        assert!(GdnKernelSet::validate_bundle(&destination).is_err());

        let mut missing = original.clone();
        missing["compilations"].as_array_mut().unwrap().remove(1);
        write_manifest(&destination, &missing);
        assert!(GdnKernelSet::validate_bundle(&destination).is_err());

        let mut escape = original.clone();
        escape["compilations"][0]["cubin"] = serde_json::json!("../sm_90/kernels.cubin");
        write_manifest(&destination, &escape);
        assert!(GdnKernelSet::validate_bundle(&destination).is_err());

        let mut source_hash = original.clone();
        source_hash["sources"]["qwen35_gdn_native.cu"] = serde_json::json!("0".repeat(64));
        write_manifest(&destination, &source_hash);
        assert!(GdnKernelSet::validate_bundle(&destination).is_err());

        let mut overridden_flags = original.clone();
        overridden_flags["flags"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!("--fmad=false"));
        write_manifest(&destination, &overridden_flags);
        assert!(GdnKernelSet::validate_bundle(&destination).is_err());

        let mut unknown_flags = original.clone();
        unknown_flags["flags"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!("--extra-device-vectorization"));
        write_manifest(&destination, &unknown_flags);
        assert!(GdnKernelSet::validate_bundle(&destination).is_err());

        write_manifest(&destination, &original);
        assert!(GdnKernelSet::validate_bundle(&destination).is_ok());
        std::fs::remove_dir_all(&destination).expect("remove temporary bundle");
    }

    #[test]
    fn aot_policy_admits_every_fresh_and_only_native_cached_shapes() {
        assert!(aot_admits(false, 1, 1));
        assert!(aot_admits(false, 255, 4));
        assert!(!aot_admits(true, 255, 4));
        assert!(aot_admits(true, 256, 4));
        assert!(!aot_admits(true, 252, 4));
        assert!(!aot_admits(true, 256, 5));
        assert!(!aot_admits(true, 511, 8));
        assert!(aot_admits(true, 512, 8));
        assert!(aot_admits(true, 256, 1));
    }

    #[test]
    fn aot_layout_covers_head_widths_and_native_grid_extents() {
        for value_heads in [32usize, 48] {
            let g = GdnGeometry::new(4, value_heads, 4).expect("geometry");
            let c = GdnCapacity::new(512, 4, 4).expect("capacity");
            let layout = GdnAotLayout::new(g, c).expect("AOT layout");
            assert_eq!(layout.packed_qk, 512 * 4 * GDN_KEY_DIM);
            assert_eq!(layout.packed_v, 512 * value_heads * GDN_KEY_DIM);
            assert_eq!(layout.gates, 512 * value_heads);
            assert_eq!(layout.state, 4 * value_heads * GDN_KEY_DIM * GDN_VALUE_DIM);
            assert_eq!(
                aot_pack_grid(512, value_heads).expect("pack grid"),
                (512, (value_heads * GDN_KEY_DIM / 256) as u32, 1)
            );
            assert_eq!(
                aot_state_tile_count(4, value_heads, GDN_KEY_DIM, GDN_VALUE_DIM)
                    .expect("state grid"),
                (4 * value_heads * 16) as u32
            );
        }
    }

    #[test]
    fn duplicate_cached_sources_still_map_final_rows_by_sequence() {
        let g = geometry();
        let sources = [1i64, 1i64];
        let plan = GdnHistoryPlan::new(g, 2, &sources, &[0, 2, 3], 3).expect("duplicate sources");
        assert_eq!(plan.initial_sequences(), 2);
        assert_eq!(
            plan.initial_elements(),
            2 * g.value_heads * g.state_row_width()
        );
        assert_eq!(
            plan.final_elements(),
            2 * g.value_heads * g.state_row_width()
        );
        assert_eq!(final_state_offset(g, 0).expect("row zero"), 0);
        assert_eq!(
            final_state_offset(g, 1).expect("row one"),
            g.value_heads * g.state_row_width()
        );
    }

    #[test]
    fn aot_capacity_rejects_state_grid_overflow_and_request_overflow() {
        let wide = GdnGeometry::new(1, 128, 1).expect("wide geometry");
        let overflowing = GdnCapacity::new(2_097_152, 2_097_152, 0).expect("capacity");
        assert!(GdnAotLayout::new(wide, overflowing).is_err());

        let g = geometry();
        let c = capacity(255, 4, 0);
        assert!(GdnPlan::new(g, c, &[0, 64, 128, 192, 256]).is_err());
        let fitting =
            GdnPlan::new(g, capacity(256, 4, 0), &[0, 64, 128, 192, 256]).expect("fitting request");
        assert_eq!(fitting.tokens(), 256);
        assert_eq!(fitting.sequences(), 4);
    }

    const AOT_BUNDLE: &str = "/tmp/oxide-qwen35-gdn-aot-pack-v1/kernels";

    fn copy_aot_bundle(destination: &std::path::Path) {
        let source = std::path::Path::new(AOT_BUNDLE);
        for relative in [
            "manifest.json",
            "NOTICE",
            "LICENSE",
            "LICENSE.candle-MIT",
            "sm_90/kernels.cubin",
            "sm_100/kernels.cubin",
            "sm_120/kernels.cubin",
        ] {
            let target = destination.join(relative);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).expect("create AOT bundle subdirectory");
            }
            std::fs::copy(source.join(relative), &target).expect("copy AOT bundle file");
        }
    }

    #[test]
    fn actual_aot_bundle_validates_and_rejects_mutated_metadata() {
        let source = std::path::Path::new(AOT_BUNDLE);
        if !source.join("manifest.json").is_file() {
            eprintln!("generated AOT packing bundle is unavailable; skipping CPU fixture");
            return;
        }
        let destination =
            std::env::temp_dir().join(format!("gdn-aot-bundle-{}-{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&destination);
        std::fs::create_dir_all(&destination).expect("create temporary AOT bundle");
        copy_aot_bundle(&destination);

        assert!(
            validate_aot_bundle(&destination).is_ok(),
            "actual three-export AOT bundle must validate"
        );
        let original: serde_json::Value = serde_json::from_slice(
            &std::fs::read(source.join("manifest.json")).expect("read AOT manifest"),
        )
        .expect("parse AOT manifest");

        let mut checksum = original.clone();
        checksum["compilations"][0]["cubin_sha256"] = serde_json::json!("0".repeat(64));
        write_manifest(&destination, &checksum);
        assert!(validate_aot_bundle(&destination).is_err());

        let mut size = original.clone();
        size["compilations"][1]["cubin_bytes"] = serde_json::json!(1);
        write_manifest(&destination, &size);
        assert!(validate_aot_bundle(&destination).is_err());

        let mut exports = original.clone();
        exports["exports"] = serde_json::json!(&AOT_REQUIRED_EXPORTS[..2]);
        write_manifest(&destination, &exports);
        assert!(validate_aot_bundle(&destination).is_err());

        let mut flags = original.clone();
        flags["flags"]
            .as_array_mut()
            .expect("flags")
            .push(serde_json::json!("--use_fast_math"));
        write_manifest(&destination, &flags);
        assert!(validate_aot_bundle(&destination).is_err());

        let mut source_hash = original.clone();
        source_hash["sources"]["qwen35_gdn_aot.cu"] = serde_json::json!("0".repeat(64));
        write_manifest(&destination, &source_hash);
        assert!(validate_aot_bundle(&destination).is_err());

        let mut escape = original.clone();
        escape["compilations"][0]["cubin"] = serde_json::json!("../sm_90/kernels.cubin");
        write_manifest(&destination, &escape);
        assert!(validate_aot_bundle(&destination).is_err());

        let mut missing = original.clone();
        missing["compilations"]
            .as_array_mut()
            .expect("compilations")
            .remove(2);
        write_manifest(&destination, &missing);
        assert!(validate_aot_bundle(&destination).is_err());

        write_manifest(&destination, &original);
        assert!(validate_aot_bundle(&destination).is_ok());
        std::fs::remove_dir_all(&destination).expect("remove temporary AOT bundle");
    }
}
