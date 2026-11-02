// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Exact OneJev block/prefix attention control for packed Qwen3.5 rows.
//!
//! Native `block_bidirectional` semantics split one logical sequence into a
//! prefix `[0, P)` and a suffix `[P, P + S)`. Prefix queries attend only the
//! first `P` keys. Suffix queries attend all `P + S` keys. Both halves are
//! noncausal, so a global noncausal launch is not a substitute: it would let
//! prefix queries observe suffix keys. This module therefore never relabels a
//! single global-noncausal call as block attention.
//!
//! This is a leaf source module. It validates a plain host descriptor plan and
//! submits retained [`AttentionBatch`] values through the existing FA4 bound
//! entry point. It owns no device allocation, provider, context, stream, or
//! completion boundary. The unsafe outer owner must retain every Q/K/V,
//! output and int32 boundary allocation plus the provider/context until one
//! request completion boundary, and must drain or quarantine the whole request
//! if an enqueue fails after an earlier call was submitted.
//!
//! The first correctness foundation emits at most two launches per sequence:
//! one prefix half and one suffix half, skipping an empty half. A request with
//! B sequences therefore costs at most 2B provider calls; B = 1 with both
//! halves present is exactly two calls. This is a correctness cost, not a
//! latency claim. A later ragged packed plan needs measured parity before it
//! can reduce the call count.

use crate::fa4::{AttentionBatch, AttentionSpec, DType, Fa4, Scale, Window};
use cuda_core::Stream;
use infinity_v3_backend_core::BackendError;

/// Native Qwen3.5 full-attention head width admitted by this leaf.
pub(crate) const BLOCK_HEAD_DIM: usize = 256;
const BLOCK_BF16_BYTES: usize = 2;
const POINTER_ALIGN: u64 = 16;
const BOUNDARY_ALIGN: u64 = 4;
const KV_HEADS: usize = 4;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide Qwen3.5 block attention: {error}"))
}

fn infer(error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide Qwen3.5 block attention: {error}"))
}

fn enqueue_error(enqueued: usize, error: impl std::fmt::Display) -> BackendError {
    if enqueued == 0 {
        infer(format!(
            "enqueue failed before any attention call was enqueued: {error}"
        ))
    } else {
        infer(format!(
            "enqueue failed after {enqueued} prior attention call(s) were enqueued; \
             the whole request must be drained or quarantined: {error}"
        ))
    }
}

/// Architecture families whose d256 FA4 provider is considered here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BlockArchitecture {
    Sm90,
    Sm100,
    Sm120,
}

impl BlockArchitecture {
    /// Translate the native `major * 10 + minor` capability encoding.
    pub(crate) fn from_capability(capability: i32) -> Result<Self, BackendError> {
        match capability {
            90 => Ok(Self::Sm90),
            100 => Ok(Self::Sm100),
            120 => Ok(Self::Sm120),
            other => Err(start(format!(
                "SM{other} is outside the supported SM90/SM100/SM120 d256 boundary"
            ))),
        }
    }

    const fn requires_shared_qk_boundaries(self) -> bool {
        matches!(self, Self::Sm100)
    }
}

/// BF16 Qwen3.5 full-attention geometry admitted by this module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlockGeometry {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl BlockGeometry {
    pub(crate) fn new(
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, BackendError> {
        if head_dim != BLOCK_HEAD_DIM {
            return Err(start(format!(
                "head dimension {head_dim} is outside the admitted d{BLOCK_HEAD_DIM} scope"
            )));
        }
        if kv_heads != KV_HEADS {
            return Err(start(format!(
                "KV head count {kv_heads} is outside the admitted qh16/qh24 kv4 scope"
            )));
        }
        if !matches!(heads, 16 | 24) {
            return Err(start(format!(
                "query head count {heads} is outside the admitted qh16/qh24 scope"
            )));
        }
        if !heads.is_multiple_of(kv_heads) {
            return Err(start("query head count is not divisible by KV head count"));
        }
        let group = heads / kv_heads;
        if !matches!(group, 4 | 6) {
            return Err(start(format!(
                "query/KV group {group} is outside the admitted group4/group6 scope"
            )));
        }
        Ok(Self {
            heads,
            kv_heads,
            head_dim,
        })
    }
}

/// One packed physical sequence: prefix tokens followed by suffix tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlockSequence {
    pub(crate) prefix: usize,
    pub(crate) length: usize,
}

impl BlockSequence {
    pub(crate) const fn new(prefix: usize, length: usize) -> Self {
        Self { prefix, length }
    }

    pub(crate) const fn suffix(&self) -> usize {
        self.length.saturating_sub(self.prefix)
    }
}

/// Plain host-visible packed allocation span. No device ownership is implied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlockTensor {
    pub(crate) address: u64,
    pub(crate) capacity_tokens: usize,
}

/// Packed BF16 tensors with the same token order as the sequence list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlockLayout {
    pub(crate) q: BlockTensor,
    pub(crate) k: BlockTensor,
    pub(crate) v: BlockTensor,
    pub(crate) out: BlockTensor,
}

/// Device int32 arrays uploaded once per changed-shape Work and retained by the
/// outer arena. Each array holds `batch + 1` entries per sequence, laid out as
/// consecutive `[0, value]` pairs; a plan uses two consecutive entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlockBoundaryBuffers {
    pub(crate) prefix: u64,
    pub(crate) suffix: u64,
    pub(crate) combined: u64,
    pub(crate) capacity_entries: usize,
}

/// Host values the outer Work uploads to the matching boundary buffers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BlockBoundaryValues {
    prefix: Vec<i32>,
    suffix: Vec<i32>,
    combined: Vec<i32>,
}

impl BlockBoundaryValues {
    pub(crate) fn prefix(&self) -> &[i32] {
        &self.prefix
    }

    pub(crate) fn suffix(&self) -> &[i32] {
        &self.suffix
    }

    pub(crate) fn combined(&self) -> &[i32] {
        &self.combined
    }
}

/// Checked cold plan. Descriptor storage is plain host metadata; all addresses
/// refer to allocations retained by the unsafe outer owner.
#[derive(Clone, Debug)]
pub(crate) struct PreparedBlockPlan {
    generation: u64,
    sequences: Vec<BlockSequence>,
    boundaries: BlockBoundaryValues,
    calls: Vec<AttentionBatch>,
}

impl PreparedBlockPlan {
    /// Validate all host layout, geometry and mask-boundary inputs once.
    ///
    /// No CUDA API is touched. The returned plan has at most `2 * sequences`
    /// descriptors, in sequence order with the prefix half before the suffix
    /// half. Empty halves are skipped.
    pub(crate) fn prepare(
        architecture: BlockArchitecture,
        spec: &AttentionSpec,
        layout: BlockLayout,
        sequences: &[BlockSequence],
        boundary_buffers: BlockBoundaryBuffers,
        generation: u64,
    ) -> Result<Self, BackendError> {
        let geometry = geometry_for(spec)?;
        if sequences.is_empty() {
            return Err(start("at least one non-empty sequence is required"));
        }

        let boundary_entries = sequences
            .len()
            .checked_mul(2)
            .ok_or_else(|| start("sequence boundary count overflow"))?;
        if boundary_buffers.capacity_entries < boundary_entries {
            return Err(start(format!(
                "boundary capacity {} is smaller than the required {boundary_entries} entries",
                boundary_buffers.capacity_entries
            )));
        }

        let mut total_tokens = 0usize;
        for (index, sequence) in sequences.iter().enumerate() {
            if sequence.length == 0 {
                return Err(start(format!("sequence {index} is empty")));
            }
            if sequence.prefix > sequence.length {
                return Err(start(format!(
                    "sequence {index} prefix {} exceeds length {}",
                    sequence.prefix, sequence.length
                )));
            }
            if sequence.length > i32::MAX as usize {
                return Err(start(format!(
                    "sequence {index} length {} exceeds the int32 boundary range",
                    sequence.length
                )));
            }
            total_tokens = total_tokens
                .checked_add(sequence.length)
                .ok_or_else(|| start("packed token count overflow"))?;
            if total_tokens > i32::MAX as usize {
                return Err(start(
                    "packed cumulative token count exceeds the int32 boundary range",
                ));
            }
        }

        let q_width = geometry
            .heads
            .checked_mul(geometry.head_dim)
            .ok_or_else(|| start("query width overflow"))?;
        let kv_width = geometry
            .kv_heads
            .checked_mul(geometry.head_dim)
            .ok_or_else(|| start("key/value width overflow"))?;
        validate_tensor("query", layout.q, q_width, total_tokens)?;
        validate_tensor("key", layout.k, kv_width, total_tokens)?;
        validate_tensor("value", layout.v, kv_width, total_tokens)?;
        validate_tensor("output", layout.out, q_width, total_tokens)?;
        validate_alias(layout, q_width, kv_width, total_tokens)?;
        for (name, address) in [
            ("prefix", boundary_buffers.prefix),
            ("suffix", boundary_buffers.suffix),
            ("combined", boundary_buffers.combined),
        ] {
            // FA4 consumes cumulative lengths as int32 scalars; unlike
            // Q/K/V/output tensors their contract is four-byte alignment.
            validate_address(name, address, BOUNDARY_ALIGN)?;
            let bytes = boundary_entries
                .checked_mul(4)
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| start(format!("{name} boundary byte extent overflow")))?;
            address
                .checked_add(bytes)
                .ok_or_else(|| start(format!("{name} boundary address range overflow")))?;
        }

        let mut prefix_values = Vec::with_capacity(boundary_entries);
        let mut suffix_values = Vec::with_capacity(boundary_entries);
        let mut combined_values = Vec::with_capacity(boundary_entries);
        for sequence in sequences {
            let prefix =
                i32::try_from(sequence.prefix).map_err(|_| start("prefix count exceeds int32"))?;
            let suffix = i32::try_from(sequence.suffix())
                .map_err(|_| start("suffix count exceeds int32"))?;
            let combined =
                i32::try_from(sequence.length).map_err(|_| start("length exceeds int32"))?;
            prefix_values.extend_from_slice(&[0, prefix]);
            suffix_values.extend_from_slice(&[0, suffix]);
            combined_values.extend_from_slice(&[0, combined]);
        }

        let head_dim = i64::try_from(geometry.head_dim)
            .map_err(|_| start("head dimension exceeds the stride range"))?;
        let q_width_i64 =
            i64::try_from(q_width).map_err(|_| start("query width exceeds the stride range"))?;
        let kv_width_i64 = i64::try_from(kv_width)
            .map_err(|_| start("key/value width exceeds the stride range"))?;
        let strides = [
            q_width_i64,
            head_dim,
            1,
            kv_width_i64,
            head_dim,
            1,
            kv_width_i64,
            head_dim,
            1,
            q_width_i64,
            head_dim,
            1,
        ];

        let mut calls = Vec::new();
        let mut sequence_start = 0usize;
        for (index, sequence) in sequences.iter().enumerate() {
            let q_start = element_address(
                layout.q.address,
                sequence_start
                    .checked_mul(q_width)
                    .ok_or_else(|| start("query element offset overflow"))?,
                "query",
            )?;
            let kv_start = element_address(
                layout.k.address,
                sequence_start
                    .checked_mul(kv_width)
                    .ok_or_else(|| start("key element offset overflow"))?,
                "key",
            )?;
            let v_start = element_address(
                layout.v.address,
                sequence_start
                    .checked_mul(kv_width)
                    .ok_or_else(|| start("value element offset overflow"))?,
                "value",
            )?;
            let out_start = element_address(
                layout.out.address,
                sequence_start
                    .checked_mul(q_width)
                    .ok_or_else(|| start("output element offset overflow"))?,
                "output",
            )?;
            let prefix_boundary = boundary_address(boundary_buffers.prefix, index)?;
            let suffix_boundary = boundary_address(boundary_buffers.suffix, index)?;
            let combined_boundary = boundary_address(boundary_buffers.combined, index)?;

            if sequence.prefix != 0 {
                calls.push(AttentionBatch {
                    q: q_start,
                    k: kv_start,
                    v: v_start,
                    out: out_start,
                    q_offsets: prefix_boundary,
                    k_offsets: prefix_boundary,
                    q_tokens: sequence.prefix,
                    k_tokens: sequence.prefix,
                    batch: 1,
                    max_q: sequence.prefix,
                    strides,
                });
            }
            if sequence.suffix() != 0 {
                let suffix_q = element_address(
                    layout.q.address,
                    sequence_start
                        .checked_add(sequence.prefix)
                        .and_then(|token| token.checked_mul(q_width))
                        .ok_or_else(|| start("suffix query element offset overflow"))?,
                    "suffix query",
                )?;
                let suffix_out = element_address(
                    layout.out.address,
                    sequence_start
                        .checked_add(sequence.prefix)
                        .and_then(|token| token.checked_mul(q_width))
                        .ok_or_else(|| start("suffix output element offset overflow"))?,
                    "suffix output",
                )?;
                // A prefix-free sequence has identical Q and K boundaries.
                // Sharing that pointer lets the stricter SM100 d256 FA4 path
                // accept the supported prefix0 case without weakening FA4.
                let q_offsets = if sequence.prefix == 0 {
                    combined_boundary
                } else {
                    suffix_boundary
                };
                calls.push(AttentionBatch {
                    q: suffix_q,
                    k: kv_start,
                    v: v_start,
                    out: suffix_out,
                    q_offsets,
                    k_offsets: combined_boundary,
                    q_tokens: sequence.suffix(),
                    k_tokens: sequence.length,
                    batch: 1,
                    max_q: sequence.suffix(),
                    strides,
                });
            }
            sequence_start = sequence_start
                .checked_add(sequence.length)
                .ok_or_else(|| start("packed sequence offset overflow"))?;
        }
        if calls.is_empty() {
            return Err(start("plan produced no attention calls"));
        }
        if architecture.requires_shared_qk_boundaries() {
            for call in &calls {
                if call.q_tokens != call.k_tokens || call.q_offsets != call.k_offsets {
                    return Err(start(
                        "SM100 d256 FA4 requires shared Q/K boundaries; a partial-prefix \
                         suffix needs distinct boundaries and is unsupported. Rejected \
                         before any enqueue",
                    ));
                }
            }
        }

        Ok(Self {
            generation,
            sequences: sequences.to_vec(),
            boundaries: BlockBoundaryValues {
                prefix: prefix_values,
                suffix: suffix_values,
                combined: combined_values,
            },
            calls,
        })
    }

    pub(crate) fn boundaries(&self) -> &BlockBoundaryValues {
        &self.boundaries
    }

    pub(crate) fn attention_calls(&self) -> &[AttentionBatch] {
        &self.calls
    }

    /// A changed shape or generation requires a fresh cold plan and boundary
    /// upload; an old plan must not be replayed.
    pub(crate) fn is_current(&self, generation: u64, sequences: &[BlockSequence]) -> bool {
        self.generation == generation && self.sequences.as_slice() == sequences
    }

    /// Submit this plan on an already-bound provider context.
    ///
    /// # Safety
    ///
    /// The FA4 provider's context must be current on the calling thread. The
    /// outer owner must retain every Q/K/V/output and boundary allocation, the
    /// provider, context and stream until one completion boundary, and must not
    /// mutate any input before that boundary. On a partial failure the whole
    /// request must be drained or quarantined.
    pub(crate) unsafe fn enqueue_bound(
        &self,
        fa4: &Fa4,
        stream: &Stream,
    ) -> Result<(), BackendError> {
        for (enqueued, call) in self.calls.iter().enumerate() {
            // SAFETY: this plan was validated cold and the caller upholds the
            // retention/context contract documented on this method.
            if let Err(error) = unsafe { fa4.forward_batch_bound(stream, call) } {
                return Err(enqueue_error(enqueued, error));
            }
        }
        Ok(())
    }
}

fn geometry_for(spec: &AttentionSpec) -> Result<BlockGeometry, BackendError> {
    if spec.dtype != DType::Bf16 {
        return Err(start("only BF16 block attention is admitted"));
    }
    if spec.causal {
        return Err(start("block attention is noncausal in both halves"));
    }
    if spec.window != Window::Global {
        return Err(start("block attention requires a global window"));
    }
    if spec.scale != Scale::RsqrtHeadDim {
        return Err(start("block attention requires rsqrt(head_dim) scaling"));
    }
    BlockGeometry::new(spec.heads, spec.kv_heads, spec.head_dim)
}

fn validate_address(name: &str, address: u64, alignment: u64) -> Result<(), BackendError> {
    if address == 0 {
        return Err(start(format!("{name} pointer is null")));
    }
    if !address.is_multiple_of(alignment) {
        return Err(start(format!(
            "{name} pointer is not {alignment}-byte aligned"
        )));
    }
    Ok(())
}

fn checked_byte_extent(tokens: usize, width: usize, name: &str) -> Result<u64, BackendError> {
    let elements = tokens
        .checked_mul(width)
        .ok_or_else(|| start(format!("{name} element extent overflow")))?;
    let bytes = elements
        .checked_mul(BLOCK_BF16_BYTES)
        .ok_or_else(|| start(format!("{name} byte extent overflow")))?;
    u64::try_from(bytes).map_err(|_| start(format!("{name} byte extent exceeds u64")))
}

fn validate_tensor(
    name: &str,
    tensor: BlockTensor,
    width: usize,
    actual_tokens: usize,
) -> Result<(), BackendError> {
    validate_address(name, tensor.address, POINTER_ALIGN)?;
    if actual_tokens > tensor.capacity_tokens {
        return Err(start(format!(
            "{name} actual token count {actual_tokens} exceeds capacity {}",
            tensor.capacity_tokens
        )));
    }
    let bytes = checked_byte_extent(tensor.capacity_tokens, width, name)?;
    tensor
        .address
        .checked_add(bytes)
        .ok_or_else(|| start(format!("{name} address range overflow")))?;
    Ok(())
}

fn validate_alias(
    layout: BlockLayout,
    q_width: usize,
    kv_width: usize,
    actual_tokens: usize,
) -> Result<(), BackendError> {
    let q_end = layout
        .q
        .address
        .checked_add(checked_byte_extent(actual_tokens, q_width, "query")?)
        .ok_or_else(|| start("query address range overflow"))?;
    let k_end = layout
        .k
        .address
        .checked_add(checked_byte_extent(actual_tokens, kv_width, "key")?)
        .ok_or_else(|| start("key address range overflow"))?;
    let v_end = layout
        .v
        .address
        .checked_add(checked_byte_extent(actual_tokens, kv_width, "value")?)
        .ok_or_else(|| start("value address range overflow"))?;
    let out_start = layout.out.address;
    let out_end = out_start
        .checked_add(checked_byte_extent(actual_tokens, q_width, "output")?)
        .ok_or_else(|| start("output address range overflow"))?;
    for (name, input_start, input_end) in [
        ("query", layout.q.address, q_end),
        ("key", layout.k.address, k_end),
        ("value", layout.v.address, v_end),
    ] {
        if out_start < input_end && input_start < out_end {
            return Err(start(format!("output storage overlaps {name} input")));
        }
    }
    Ok(())
}

fn element_address(base: u64, elements: usize, name: &str) -> Result<u64, BackendError> {
    let bytes = elements
        .checked_mul(BLOCK_BF16_BYTES)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| start(format!("{name} element byte offset overflow")))?;
    let address = base
        .checked_add(bytes)
        .ok_or_else(|| start(format!("{name} address overflow")))?;
    validate_address(name, address, POINTER_ALIGN)?;
    Ok(address)
}

fn boundary_address(base: u64, index: usize) -> Result<u64, BackendError> {
    let bytes = index
        .checked_mul(2)
        .and_then(|entries| entries.checked_mul(4))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| start("boundary pointer offset overflow"))?;
    let address = base
        .checked_add(bytes)
        .ok_or_else(|| start("boundary address overflow"))?;
    validate_address("sequence boundary", address, BOUNDARY_ALIGN)?;
    Ok(address)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(heads: usize) -> AttentionSpec {
        AttentionSpec {
            heads,
            kv_heads: 4,
            head_dim: BLOCK_HEAD_DIM,
            causal: false,
            window: Window::Global,
            dtype: DType::Bf16,
            scale: Scale::RsqrtHeadDim,
        }
    }

    fn layout(capacity_tokens: usize) -> BlockLayout {
        BlockLayout {
            q: BlockTensor {
                address: 0x10_0000,
                capacity_tokens,
            },
            k: BlockTensor {
                address: 0x20_0000,
                capacity_tokens,
            },
            v: BlockTensor {
                address: 0x30_0000,
                capacity_tokens,
            },
            out: BlockTensor {
                address: 0x40_0000,
                capacity_tokens,
            },
        }
    }

    fn boundaries(capacity_entries: usize) -> BlockBoundaryBuffers {
        BlockBoundaryBuffers {
            prefix: 0x50_0000,
            suffix: 0x51_0000,
            combined: 0x52_0000,
            capacity_entries,
        }
    }

    fn prepare(
        architecture: BlockArchitecture,
        sequences: &[BlockSequence],
        layout: BlockLayout,
        boundaries: BlockBoundaryBuffers,
        generation: u64,
    ) -> Result<PreparedBlockPlan, BackendError> {
        PreparedBlockPlan::prepare(
            architecture,
            &spec(16),
            layout,
            sequences,
            boundaries,
            generation,
        )
    }

    fn sequence_start(sequences: &[BlockSequence], index: usize) -> usize {
        sequences[..index]
            .iter()
            .map(|sequence| sequence.length)
            .sum()
    }

    #[test]
    fn packed_int32_boundaries_accept_every_four_byte_residue() {
        let sequence = [BlockSequence::new(359, 460)];
        // Each actual 512-row tensor is several MiB, unlike the small-shape
        // fixture below. Space the simulated allocations far enough apart.
        let mut storage = layout(512);
        storage.q.address = 0x1000_0000;
        storage.k.address = 0x2000_0000;
        storage.v.address = 0x3000_0000;
        storage.out.address = 0x4000_0000;
        for residue in [0, 4, 8, 12] {
            let mut buffers = boundaries(2);
            buffers.prefix += residue;
            buffers.suffix += (residue + 8) % 16;
            buffers.combined += (residue + 4) % 16;
            let plan = prepare(BlockArchitecture::Sm90, &sequence, storage, buffers, 1).unwrap();
            let calls = plan.attention_calls();
            assert_eq!(calls[0].q_offsets, buffers.prefix);
            assert_eq!(calls[0].k_offsets, buffers.prefix);
            assert_eq!(calls[1].q_offsets, buffers.suffix);
            assert_eq!(calls[1].k_offsets, buffers.combined);
        }
        let mut invalid = boundaries(2);
        invalid.prefix += 2;
        assert!(prepare(BlockArchitecture::Sm90, &sequence, storage, invalid, 1).is_err());
        let mut tensor = storage;
        tensor.q.address += 4;
        assert!(prepare(BlockArchitecture::Sm90, &sequence, tensor, boundaries(2), 1).is_err());
    }

    #[test]
    fn single_sequence_prefix0_full_and_partial_have_exact_halves() {
        for heads in [16, 24] {
            let q_width = heads * BLOCK_HEAD_DIM;
            let spec = spec(heads);
            let layout = layout(16);

            let empty_prefix = [BlockSequence::new(0, 5)];
            let plan = PreparedBlockPlan::prepare(
                BlockArchitecture::Sm120,
                &spec,
                layout,
                &empty_prefix,
                boundaries(16),
                3,
            )
            .unwrap();
            assert_eq!(plan.attention_calls().len(), 1);
            let call = plan.attention_calls()[0];
            assert_eq!((call.q_tokens, call.k_tokens), (5, 5));
            assert_eq!((call.batch, call.max_q), (1, 5));
            assert_eq!(call.q_offsets, call.k_offsets);
            assert_eq!(call.q, layout.q.address);
            assert_eq!(call.k, layout.k.address);
            assert_eq!(call.out, layout.out.address);
            assert_eq!(plan.boundaries().prefix(), &[0, 0]);
            assert_eq!(plan.boundaries().suffix(), &[0, 5]);
            assert_eq!(plan.boundaries().combined(), &[0, 5]);

            let full_prefix = [BlockSequence::new(5, 5)];
            let plan = PreparedBlockPlan::prepare(
                BlockArchitecture::Sm120,
                &spec,
                layout,
                &full_prefix,
                boundaries(16),
                4,
            )
            .unwrap();
            assert_eq!(plan.attention_calls().len(), 1);
            let call = plan.attention_calls()[0];
            assert_eq!((call.q_tokens, call.k_tokens), (5, 5));
            assert_eq!(call.q_offsets, call.k_offsets);
            assert_eq!(call.q_offsets, boundaries(16).prefix);
            assert_eq!(plan.boundaries().suffix(), &[0, 0]);

            let partial = [BlockSequence::new(3, 8)];
            let plan = PreparedBlockPlan::prepare(
                BlockArchitecture::Sm120,
                &spec,
                layout,
                &partial,
                boundaries(16),
                5,
            )
            .unwrap();
            assert_eq!(plan.attention_calls().len(), 2);
            let prefix = plan.attention_calls()[0];
            let suffix = plan.attention_calls()[1];
            assert_eq!((prefix.q_tokens, prefix.k_tokens), (3, 3));
            assert_eq!((suffix.q_tokens, suffix.k_tokens), (5, 8));
            assert_eq!(prefix.q_offsets, prefix.k_offsets);
            assert_ne!(suffix.q_offsets, suffix.k_offsets);
            assert_eq!(prefix.out, layout.out.address);
            assert_eq!(
                suffix.q,
                layout.q.address + u64::try_from(3 * 2 * q_width).unwrap()
            );
            assert_eq!(suffix.k, layout.k.address);
            assert_eq!(
                suffix.out,
                layout.out.address + u64::try_from(3 * 2 * q_width).unwrap()
            );
            assert_eq!(suffix.v, layout.v.address);
            assert_eq!(plan.boundaries().prefix(), &[0, 3]);
            assert_eq!(plan.boundaries().suffix(), &[0, 5]);
            assert_eq!(plan.boundaries().combined(), &[0, 8]);
        }
    }

    #[test]
    fn ragged_mixed_lengths_and_prefixes_keep_per_sequence_offsets() {
        let sequences = [
            BlockSequence::new(3, 5),
            BlockSequence::new(0, 4),
            BlockSequence::new(7, 7),
            BlockSequence::new(2, 9),
        ];
        let layout = layout(32);
        let plan = PreparedBlockPlan::prepare(
            BlockArchitecture::Sm120,
            &spec(24),
            layout,
            &sequences,
            boundaries(16),
            11,
        )
        .unwrap();
        assert_eq!(plan.attention_calls().len(), 6);
        let q_width = 24 * BLOCK_HEAD_DIM;
        let kv_width = 4 * BLOCK_HEAD_DIM;
        assert_eq!(plan.boundaries().prefix(), &[0, 3, 0, 0, 0, 7, 0, 2]);
        assert_eq!(plan.boundaries().suffix(), &[0, 2, 0, 4, 0, 0, 0, 7]);
        assert_eq!(plan.boundaries().combined(), &[0, 5, 0, 4, 0, 7, 0, 9]);

        let mut call_index = 0;
        for (index, sequence) in sequences.iter().enumerate() {
            let start = sequence_start(&sequences, index);
            if sequence.prefix != 0 {
                let call = plan.attention_calls()[call_index];
                assert_eq!(call.q, layout.q.address + (start * q_width * 2) as u64);
                assert_eq!(call.k, layout.k.address + (start * kv_width * 2) as u64);
                assert_eq!(
                    (call.q_tokens, call.k_tokens),
                    (sequence.prefix, sequence.prefix)
                );
                call_index += 1;
            }
            if sequence.suffix() != 0 {
                let call = plan.attention_calls()[call_index];
                assert_eq!(
                    call.q,
                    layout.q.address + ((start + sequence.prefix) * q_width * 2) as u64
                );
                assert_eq!(call.k, layout.k.address + (start * kv_width * 2) as u64);
                assert_eq!(
                    (call.q_tokens, call.k_tokens),
                    (sequence.suffix(), sequence.length)
                );
                call_index += 1;
            }
        }
        assert_eq!(call_index, 6);
    }

    #[test]
    fn changed_shape_or_generation_invalidates_the_old_plan() {
        let first = [BlockSequence::new(2, 6)];
        let second = [BlockSequence::new(2, 7)];
        let plan = prepare(
            BlockArchitecture::Sm120,
            &first,
            layout(16),
            boundaries(16),
            21,
        )
        .unwrap();
        assert!(plan.is_current(21, &first));
        assert!(!plan.is_current(21, &second));
        assert!(!plan.is_current(22, &first));
    }

    #[test]
    fn rejects_invalid_sequences_layout_alignment_ranges_and_aliases() {
        let valid = [BlockSequence::new(1, 3)];
        assert!(prepare(BlockArchitecture::Sm120, &[], layout(8), boundaries(8), 1).is_err());
        assert!(prepare(
            BlockArchitecture::Sm120,
            &[BlockSequence::new(0, 0)],
            layout(8),
            boundaries(8),
            1
        )
        .is_err());
        assert!(prepare(
            BlockArchitecture::Sm120,
            &[BlockSequence::new(4, 3)],
            layout(8),
            boundaries(8),
            1
        )
        .is_err());
        assert!(prepare(
            BlockArchitecture::Sm120,
            &[BlockSequence::new(0, 9)],
            layout(8),
            boundaries(8),
            1
        )
        .is_err());
        assert!(prepare(
            BlockArchitecture::Sm120,
            &[BlockSequence::new(0, (i32::MAX as usize) + 1)],
            layout((i32::MAX as usize) + 1),
            boundaries(8),
            1
        )
        .is_err());
        assert!(prepare(
            BlockArchitecture::Sm120,
            &valid,
            layout(8),
            boundaries(1),
            1
        )
        .is_err());

        let mut misaligned = layout(8);
        misaligned.q.address += 8;
        assert!(prepare(
            BlockArchitecture::Sm120,
            &valid,
            misaligned,
            boundaries(8),
            1
        )
        .is_err());

        let mut misaligned_boundary = boundaries(8);
        misaligned_boundary.prefix += 2;
        assert!(prepare(
            BlockArchitecture::Sm120,
            &valid,
            layout(8),
            misaligned_boundary,
            1
        )
        .is_err());

        let mut overflow = layout(8);
        overflow.out.address = u64::MAX - (POINTER_ALIGN - 1);
        assert!(prepare(BlockArchitecture::Sm120, &valid, overflow, boundaries(8), 1).is_err());

        let mut aliased = layout(8);
        aliased.out.address = aliased.q.address;
        assert!(prepare(BlockArchitecture::Sm120, &valid, aliased, boundaries(8), 1).is_err());

        let mut aliased_middle = layout(8);
        aliased_middle.out.address = 0x10_0000 + 32;
        assert!(prepare(
            BlockArchitecture::Sm120,
            &valid,
            aliased_middle,
            boundaries(8),
            1
        )
        .is_err());

        for (heads, kv_heads, head_dim) in [
            (12, 4, 256),
            (32, 4, 256),
            (16, 8, 256),
            (24, 4, 128),
            (0, 4, 256),
        ] {
            assert!(BlockGeometry::new(heads, kv_heads, head_dim).is_err());
        }
        assert_eq!(BlockGeometry::new(16, 4, 256).unwrap().heads, 16);
        assert_eq!(BlockGeometry::new(24, 4, 256).unwrap().heads, 24);
    }

    #[test]
    fn sm100_rejects_partial_prefix_but_accepts_prefix0_and_full() {
        for heads in [16, 24] {
            let spec = spec(heads);
            for architecture in [BlockArchitecture::Sm90, BlockArchitecture::Sm120] {
                assert!(PreparedBlockPlan::prepare(
                    architecture,
                    &spec,
                    layout(16),
                    &[BlockSequence::new(3, 8)],
                    boundaries(16),
                    1,
                )
                .is_ok());
            }
            assert!(PreparedBlockPlan::prepare(
                BlockArchitecture::Sm100,
                &spec,
                layout(16),
                &[BlockSequence::new(3, 8)],
                boundaries(16),
                1,
            )
            .is_err());
            assert!(PreparedBlockPlan::prepare(
                BlockArchitecture::Sm100,
                &spec,
                layout(16),
                &[BlockSequence::new(0, 8)],
                boundaries(16),
                1,
            )
            .is_ok());
            assert!(PreparedBlockPlan::prepare(
                BlockArchitecture::Sm100,
                &spec,
                layout(16),
                &[BlockSequence::new(8, 8)],
                boundaries(16),
                1,
            )
            .is_ok());
        }
        assert!(BlockArchitecture::from_capability(80).is_err());
        assert_eq!(
            BlockArchitecture::from_capability(100).unwrap(),
            BlockArchitecture::Sm100
        );
    }

    #[test]
    fn descriptor_mask_coverage_has_no_cross_sequence_leakage() {
        let sequences = [
            BlockSequence::new(0, 3),
            BlockSequence::new(2, 5),
            BlockSequence::new(4, 7),
            BlockSequence::new(3, 9),
        ];
        let layout = layout(32);
        let q_width = 16 * BLOCK_HEAD_DIM;
        let kv_width = 4 * BLOCK_HEAD_DIM;
        let plan = PreparedBlockPlan::prepare(
            BlockArchitecture::Sm120,
            &spec(16),
            layout,
            &sequences,
            boundaries(16),
            1,
        )
        .unwrap();
        let token_index = |base: u64, address: u64, width: usize| {
            usize::try_from((address - base) / u64::try_from(width * 2).unwrap()).unwrap()
        };
        for (index, sequence) in sequences.iter().enumerate() {
            let start = sequence_start(&sequences, index);
            for row in 0..sequence.length {
                let global_row = start + row;
                let mut covering = plan.attention_calls().iter().filter(|call| {
                    let q_start = token_index(layout.q.address, call.q, q_width);
                    global_row >= q_start && global_row < q_start + call.q_tokens
                });
                let call = covering.next().expect("each row has one attention call");
                assert!(covering.next().is_none(), "row belongs to exactly one call");
                let k_start = token_index(layout.k.address, call.k, kv_width);
                let expected = if row < sequence.prefix {
                    sequence.prefix
                } else {
                    sequence.length
                };
                assert_eq!(k_start, start, "keys start at this sequence only");
                assert_eq!(
                    k_start + call.k_tokens,
                    start + expected,
                    "mask covers exactly the native block limit"
                );
                for key in k_start..k_start + call.k_tokens {
                    assert!(
                        key >= start && key < start + sequence.length,
                        "no cross-sequence key leakage"
                    );
                }
            }
        }
    }
}
