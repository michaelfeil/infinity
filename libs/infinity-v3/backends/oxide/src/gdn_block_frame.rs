// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! B1 OneJev block frames for the prepared GDN operator. One prefix frame runs
//! the fresh AOT policy and writes the next layer's F32 state directly into the
//! outer Work allocation. One optional suffix frame consumes that state through
//! the existing cached policy. Auxiliary metadata is already resident in Work
//! uploads; this leaf stages no host metadata, uploads nothing and allocates
//! nothing. Frame tokens borrow the outer WorkBlockPlan for their lifetime.

use super::*;
use std::mem::size_of;

/// Device metadata views packed once by the outer Work block upload.
///
/// CUs and history CUs are `u32` words. The source map is two `u32` words that
/// the existing kernels reinterpret as one aligned little-endian `i64`.
#[derive(Clone, Copy)]
#[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 B1 block integration.
pub(crate) struct GdnBlockMetadata<'a> {
    pub(crate) prefix_cu: GdnBufferView<'a, u32>,
    pub(crate) suffix_cu: GdnBufferView<'a, u32>,
    pub(crate) history_cu: GdnBufferView<'a, u32>,
    pub(crate) sources: GdnBufferView<'a, u32>,
}

/// Optional suffix request. Its state must be the cached form.
#[derive(Clone, Copy)]
#[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 B1 block integration.
pub(crate) struct GdnBlockSuffix<'a> {
    pub(crate) inputs: GdnInputViews<'a>,
    pub(crate) cumulative: &'a [u32],
    pub(crate) state: GdnStateViews<'a>,
    pub(crate) final_state: Option<GdnBufferView<'a, f32>>,
    pub(crate) output: GdnBufferView<'a, bf16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GdnBlockFrameKey {
    owner: usize,
    metadata_generation: u64,
    slot: u8,
    tokens: i32,
    sequences: i32,
    cached: bool,
    cumulative_len: usize,
    history_cu_len: usize,
    sources_len: usize,
    write_final_state: bool,
}

#[derive(Clone, Copy)]
struct GdnBlockFrameMetadata<'a> {
    cumulative: GdnBufferView<'a, u32>,
    history_cu: Option<GdnBufferView<'a, u32>>,
    sources: Option<GdnBufferView<'a, u32>>,
}

/// Independently valid copy token for one slot of one prepared pair.
#[derive(Clone, Copy)]
#[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 B1 block integration.
pub(crate) struct GdnBlockFrame<'a> {
    key: GdnBlockFrameKey,
    inputs: GdnInputViews<'a>,
    cumulative: &'a [u32],
    state: GdnStateViews<'a>,
    metadata: GdnBlockFrameMetadata<'a>,
    final_state: Option<GdnBufferView<'a, f32>>,
    output: GdnBufferView<'a, bf16>,
    write_final_state: bool,
}

/// Prefix slot 0 plus the optional suffix slot 1.
#[derive(Clone, Copy)]
#[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 B1 block integration.
pub(crate) struct GdnBlockPair<'a> {
    pub(crate) prefix: GdnBlockFrame<'a>,
    pub(crate) suffix: Option<GdnBlockFrame<'a>>,
}

/// Per-frame enqueue result. Both borrows track the frame's outer Work targets,
/// never the temporary `&mut self` borrow of the prepared scratch.
#[derive(Clone, Copy)]
#[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 B1 block integration.
pub(crate) struct GdnBlockOutput<'f> {
    pub(crate) output: GdnBufferView<'f, bf16>,
    pub(crate) final_state: Option<GdnBufferView<'f, f32>>,
}

fn not_queued(error: BackendError) -> GdnOwnedFailure {
    GdnOwnedFailure {
        error,
        queued: false,
    }
}

fn validate_cumulative_shape(cumulative: &[u32], name: &str) -> Result<(), BackendError> {
    if cumulative.len() != 2 {
        return Err(infer(format!(
            "{name} must contain exactly one B1 sequence boundary pair"
        )));
    }
    if cumulative[0] != 0 || cumulative[1] == 0 {
        return Err(infer(format!(
            "{name} must be the nonzero increasing pair [0, tokens]"
        )));
    }
    Ok(())
}

fn validate_metadata_view(
    core: &GdnCore,
    view: GdnBufferView<'_, u32>,
    expected_words: usize,
    alignment: usize,
    name: &str,
) -> Result<(), BackendError> {
    if view.len() != expected_words {
        return Err(infer(format!(
            "{name} has {} device words, expected {expected_words}",
            view.len()
        )));
    }
    require_context(&core.stream, view.context(), name)?;
    let pointer = view.cu_deviceptr();
    if pointer == 0 || !pointer_alignment(pointer).is_multiple_of(alignment) {
        return Err(infer(format!(
            "{name} pointer is not {alignment}-byte aligned"
        )));
    }
    Ok(())
}

fn validate_source_view(
    core: &GdnCore,
    view: GdnBufferView<'_, u32>,
    sources: usize,
    name: &str,
) -> Result<(), BackendError> {
    let words = sources
        .checked_mul(2)
        .ok_or_else(|| infer("source metadata word count overflows"))?;
    validate_metadata_view(core, view, words, 8, name)
}

fn device_range(
    pointer: u64,
    elements: usize,
    element_size: usize,
    name: &str,
) -> Result<(u64, u64), BackendError> {
    if elements == 0 {
        return Ok((pointer, pointer));
    }
    if pointer == 0 || element_size == 0 {
        return Err(infer(format!("{name} has no addressable device storage")));
    }
    let bytes = elements
        .checked_mul(element_size)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| infer(format!("{name} byte extent overflows")))?;
    let end = pointer
        .checked_add(bytes)
        .ok_or_else(|| infer(format!("{name} device range overflows")))?;
    Ok((pointer, end))
}

fn view_range<T>(view: GdnBufferView<'_, T>, name: &str) -> Result<(u64, u64), BackendError> {
    device_range(view.cu_deviceptr(), view.len(), size_of::<T>(), name)
}

fn buffer_range<T>(buffer: &DeviceBuffer<T>, name: &str) -> Result<(u64, u64), BackendError> {
    device_range(buffer.cu_deviceptr(), buffer.len(), size_of::<T>(), name)
}

fn require_disjoint(target: (u64, u64), other: (u64, u64), name: &str) -> Result<(), BackendError> {
    if target.0 < other.1 && other.0 < target.1 {
        return Err(infer(format!("GDN block target aliases {name} byte range")));
    }
    Ok(())
}

fn require_adjacent_outputs(prefix: (u64, u64), suffix: (u64, u64)) -> Result<(), BackendError> {
    if prefix.1 != suffix.0 {
        return Err(infer(
            "GDN block prefix/suffix outputs must be ordered adjacent spans",
        ));
    }
    require_disjoint(prefix, suffix, "combined prefix/suffix output")
}

fn require_disjoint_view<T>(
    target: (u64, u64),
    view: GdnBufferView<'_, T>,
    name: &str,
) -> Result<(), BackendError> {
    require_disjoint(target, view_range(view, name)?, name)
}

fn require_disjoint_buffer<T>(
    target: (u64, u64),
    buffer: &DeviceBuffer<T>,
    name: &str,
) -> Result<(), BackendError> {
    require_disjoint(target, buffer_range(buffer, name)?, name)
}

/// An external target must be wholly outside every input, history row and owned
/// scratch allocation it could be overwritten by or overwrite.
#[allow(clippy::too_many_arguments)]
fn validate_target_disjoint<T>(
    core: &GdnCore,
    inputs: &GdnInputViews<'_>,
    state: GdnStateViews<'_>,
    target: GdnBufferView<'_, T>,
    target_name: &str,
    metadata: &[GdnBufferView<'_, u32>],
    initial: Option<GdnBufferView<'_, f32>>,
    externals: &[GdnBufferView<'_, f32>],
) -> Result<(), BackendError> {
    let range = view_range(target, target_name)?;
    require_disjoint_view(range, inputs.qkv, "qkv input")?;
    require_disjoint_view(range, inputs.z, "z input")?;
    require_disjoint_view(range, inputs.ab, "a/b input")?;
    if let GdnStateViews::Cached { history_rows, .. } = state {
        require_disjoint_view(range, history_rows, "convolution history")?;
    }
    if let Some(initial) = initial {
        require_disjoint_view(range, initial, "cached initial state")?;
    }
    for external in externals {
        require_disjoint_view(range, *external, "external state")?;
    }
    for view in metadata {
        require_disjoint_view(range, *view, "block metadata")?;
    }
    validate_owned_scratch_disjoint(core, range)
}

/// Cold pair admission checks external metadata against every owned scratch span.
fn validate_owned_scratch_disjoint(core: &GdnCore, range: (u64, u64)) -> Result<(), BackendError> {
    require_disjoint_buffer(range, &core.convolution, "convolution scratch")?;
    require_disjoint_buffer(range, &core.qk, "qk scratch")?;
    require_disjoint_buffer(range, &core.decay_beta, "decay/beta scratch")?;
    require_disjoint_buffer(range, &core.recurrence, "recurrence scratch")?;
    require_disjoint_buffer(range, &core.output, "output scratch")?;
    require_disjoint_buffer(range, &core.final_state, "core final state")?;
    require_disjoint_buffer(range, &core.cu, "cumulative metadata")?;
    require_disjoint_buffer(range, &core.sources, "source metadata")?;
    require_disjoint_buffer(range, &core.history_cu, "history metadata")?;
    if let Some(work) = core.aot.as_ref() {
        require_disjoint_buffer(range, &work.packed_q, "AOT packed Q")?;
        require_disjoint_buffer(range, &work.packed_k, "AOT packed K")?;
        require_disjoint_buffer(range, &work.packed_v, "AOT packed V")?;
        require_disjoint_buffer(range, &work.alpha, "AOT alpha")?;
        require_disjoint_buffer(range, &work.beta, "AOT beta")?;
        require_disjoint_buffer(range, &work.state_vk, "AOT state")?;
        require_disjoint_buffer(range, &work.initial_vk, "AOT initial state")?;
        require_disjoint_buffer(range, &work.workspace, "AOT workspace")?;
    }
    Ok(())
}

fn validate_final_target(
    core: &GdnCore,
    inputs: &GdnInputViews<'_>,
    state: GdnStateViews<'_>,
    target: GdnBufferView<'_, f32>,
    plan: &GdnSubmitPlan,
    metadata: &[GdnBufferView<'_, u32>],
) -> Result<(), BackendError> {
    let expected = plan
        .final_elements
        .ok_or_else(|| infer("external final state was supplied without a write request"))?;
    if target.len() != expected {
        return Err(infer(format!(
            "external final state has {} elements, expected {expected}",
            target.len()
        )));
    }
    require_context(&core.stream, target.context(), "external final state")?;
    if pointer_alignment(target.cu_deviceptr()) < 16 {
        return Err(infer(
            "external final state base alignment is below sixteen bytes",
        ));
    }
    let initial = match state {
        GdnStateViews::Fresh => None,
        GdnStateViews::Cached { initial, .. } => Some(initial),
    };
    validate_target_disjoint(
        core,
        inputs,
        state,
        target,
        "external final state",
        metadata,
        initial,
        &[],
    )
}

fn validate_output_target(
    core: &GdnCore,
    plan: &GdnSubmitPlan,
    inputs: &GdnInputViews<'_>,
    state: GdnStateViews<'_>,
    target: GdnBufferView<'_, bf16>,
    metadata: &[GdnBufferView<'_, u32>],
    externals: &[GdnBufferView<'_, f32>],
) -> Result<(), BackendError> {
    if target.len() != plan.output_elements {
        return Err(infer(format!(
            "GDN block output target has {} elements, expected {}",
            target.len(),
            plan.output_elements
        )));
    }
    require_context(&core.stream, target.context(), "GDN block output target")?;
    if pointer_alignment(target.cu_deviceptr()) < 2 {
        return Err(infer(
            "GDN block output target base alignment is below two bytes",
        ));
    }
    let initial = match state {
        GdnStateViews::Fresh => None,
        GdnStateViews::Cached { initial, .. } => Some(initial),
    };
    validate_target_disjoint(
        core,
        inputs,
        state,
        target,
        "GDN block output target",
        metadata,
        initial,
        externals,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_block_frame_key(
    key: &GdnBlockFrameKey,
    owner: usize,
    generation: u64,
    cumulative: &[u32],
    cached: bool,
    history_cu_len: usize,
    sources_len: usize,
    write_final_state: bool,
) -> Result<(), BackendError> {
    if key.owner != owner {
        return Err(infer(
            "GDN block frame token belongs to a different prepared owner",
        ));
    }
    if key.metadata_generation != generation {
        return Err(infer("GDN block frame metadata generation is stale"));
    }
    if key.slot > 1 {
        return Err(infer("GDN block frame slot is outside 0/1"));
    }
    let expected_slot = u8::from(cached);
    if key.slot != expected_slot {
        return Err(infer("GDN block frame slot does not match its state kind"));
    }
    if key.cached != cached {
        return Err(infer("GDN block frame cached mode changed"));
    }
    if key.cumulative_len != cumulative.len() {
        return Err(infer("GDN block frame cumulative metadata changed"));
    }
    if !cached {
        if key.history_cu_len != 0 || key.sources_len != 0 {
            return Err(infer(
                "GDN block frame fresh token carries history metadata",
            ));
        }
    } else if key.history_cu_len != history_cu_len || key.sources_len != sources_len {
        return Err(infer("GDN block frame cached history metadata changed"));
    }
    if key.write_final_state != write_final_state {
        return Err(infer("GDN block frame final-state policy changed"));
    }
    Ok(())
}

impl GdnPrepared {
    /// Admit one fresh prefix and one optional cached suffix without any launch,
    /// host staging or device upload. The one generation increment invalidates
    /// every earlier standalone frame and every earlier block pair.
    ///
    /// # Safety
    /// The outer WorkBlockPlan owns the stream, all metadata, input/history
    /// views, external state and combined output targets, and every retained
    /// model/slab allocation through stream completion. The supplied metadata
    /// must already be uploaded on that same stream. No owner may be reused or
    /// mutated while queued work can access it. On any error the caller must
    /// drain or quarantine the whole stream, including earlier layers. This call
    /// enqueues no work.
    #[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 B1 block integration.
    pub(crate) unsafe fn prepare_block_pair<'a>(
        &mut self,
        prefix_inputs: GdnInputViews<'a>,
        prefix_cumulative: &'a [u32],
        suffix: Option<GdnBlockSuffix<'a>>,
        metadata: GdnBlockMetadata<'a>,
        prefix_final_state: GdnBufferView<'a, f32>,
        prefix_output: GdnBufferView<'a, bf16>,
    ) -> Result<GdnBlockPair<'a>, GdnOwnedFailure> {
        validate_cumulative_shape(prefix_cumulative, "prefix cumulative metadata")
            .map_err(not_queued)?;
        let prefix_plan = self
            .preflight_views(
                &prefix_inputs,
                prefix_cumulative,
                GdnStateViews::Fresh,
                true,
            )
            .map_err(not_queued)?;
        if !prefix_plan.aot.as_ref().is_some_and(|aot| aot.write_final) {
            return Err(not_queued(infer(
                "GDN block prefix requires the fresh AOT final-state path",
            )));
        }
        validate_metadata_view(
            self.core_ref().map_err(not_queued)?,
            metadata.prefix_cu,
            prefix_cumulative.len(),
            4,
            "prefix cumulative metadata",
        )
        .map_err(not_queued)?;
        {
            let core = self.core_ref().map_err(not_queued)?;
            validate_final_target(
                core,
                &prefix_inputs,
                GdnStateViews::Fresh,
                prefix_final_state,
                &prefix_plan,
                &[metadata.prefix_cu],
            )
            .map_err(not_queued)?;
        }

        let suffix_parts = match suffix {
            None => None,
            Some(suffix) => {
                validate_cumulative_shape(suffix.cumulative, "suffix cumulative metadata")
                    .map_err(not_queued)?;
                if !matches!(suffix.state, GdnStateViews::Cached { .. }) {
                    return Err(not_queued(infer(
                        "GDN block suffix requires the cached state form",
                    )));
                }
                let write_final_state = suffix.final_state.is_some();
                let suffix_plan = self
                    .preflight_views(
                        &suffix.inputs,
                        suffix.cumulative,
                        suffix.state,
                        write_final_state,
                    )
                    .map_err(not_queued)?;
                let (history_cumulative, sources, initial, history_rows) = match suffix.state {
                    GdnStateViews::Cached {
                        initial,
                        history_rows,
                        history_cumulative,
                        sources,
                    } => (history_cumulative, sources, initial, history_rows),
                    GdnStateViews::Fresh => {
                        return Err(not_queued(infer(
                            "GDN block suffix requires the cached state form",
                        )))
                    }
                };
                let channels = self.geometry.channels();
                if prefix_inputs.qkv_row_stride != channels
                    || suffix.inputs.qkv_row_stride != channels
                {
                    return Err(not_queued(infer(
                        "GDN block qkv row stride must equal the channel width",
                    )));
                }
                if initial.len() != prefix_final_state.len()
                    || initial.cu_deviceptr() != prefix_final_state.cu_deviceptr()
                {
                    return Err(not_queued(infer(
                        "GDN block suffix initial must be the exact prefix final state",
                    )));
                }
                let expected_history_rows = prefix_plan.tokens.min(3);
                let history_end = expected_history_rows;
                if history_cumulative != &[0i32, history_end][..]
                    || sources.first().copied() != Some(0)
                {
                    return Err(not_queued(infer(
                        "GDN block B1 history must be [0, min(P, 3)] with source 0",
                    )));
                }
                let prefix_tokens = usize::try_from(prefix_plan.tokens)
                    .map_err(|_| not_queued(infer("GDN block prefix token count is negative")))?;
                let expected_history_rows = usize::try_from(expected_history_rows)
                    .map_err(|_| not_queued(infer("GDN block history row count is negative")))?;
                let history_offset = prefix_tokens
                    .checked_sub(expected_history_rows)
                    .and_then(|rows| rows.checked_mul(channels))
                    .and_then(|elements| elements.checked_mul(size_of::<bf16>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or_else(|| not_queued(infer("GDN block history range overflows")))?;
                let expected_pointer = prefix_inputs
                    .qkv
                    .cu_deviceptr()
                    .checked_add(history_offset)
                    .ok_or_else(|| not_queued(infer("GDN block history range overflows")))?;
                let expected_elements = expected_history_rows
                    .checked_mul(channels)
                    .ok_or_else(|| not_queued(infer("GDN block history range overflows")))?;
                if history_rows.cu_deviceptr() != expected_pointer
                    || history_rows.len() != expected_elements
                {
                    return Err(not_queued(infer(
                        "GDN block history rows must be the last min(P, 3) raw prefix qkv rows",
                    )));
                }
                let core = self.core_ref().map_err(not_queued)?;
                validate_metadata_view(
                    core,
                    metadata.suffix_cu,
                    suffix.cumulative.len(),
                    4,
                    "suffix cumulative metadata",
                )
                .map_err(not_queued)?;
                validate_metadata_view(
                    core,
                    metadata.history_cu,
                    history_cumulative.len(),
                    4,
                    "history cumulative metadata",
                )
                .map_err(not_queued)?;
                validate_source_view(core, metadata.sources, sources.len(), "source map")
                    .map_err(not_queued)?;
                let metadata_views = [metadata.suffix_cu, metadata.history_cu, metadata.sources];
                validate_target_disjoint(
                    core,
                    &suffix.inputs,
                    suffix.state,
                    initial,
                    "cached initial state",
                    &metadata_views,
                    None,
                    &[],
                )
                .map_err(not_queued)?;
                if let Some(target) = suffix.final_state {
                    validate_final_target(
                        core,
                        &suffix.inputs,
                        suffix.state,
                        target,
                        &suffix_plan,
                        &metadata_views,
                    )
                    .map_err(not_queued)?;
                }
                Some((suffix, suffix_plan))
            }
        };

        {
            let core = self.core_ref().map_err(not_queued)?;
            let mut externals = [prefix_final_state, prefix_final_state];
            let mut external_count = 1;
            if let Some((suffix, _)) = &suffix_parts {
                if let Some(target) = suffix.final_state {
                    externals[external_count] = target;
                    external_count += 1;
                }
            }
            let (prefix_metadata, prefix_metadata_len) = (
                [
                    metadata.prefix_cu,
                    metadata.suffix_cu,
                    metadata.history_cu,
                    metadata.sources,
                ],
                if suffix_parts.is_some() { 4 } else { 1 },
            );
            // These immutable metadata views are admitted once per outer
            // request, before either frame can enqueue. No per-layer scan.
            for view in &prefix_metadata[..prefix_metadata_len] {
                let range = view_range(*view, "block metadata").map_err(not_queued)?;
                validate_owned_scratch_disjoint(core, range).map_err(not_queued)?;
            }
            validate_output_target(
                core,
                &prefix_plan,
                &prefix_inputs,
                GdnStateViews::Fresh,
                prefix_output,
                &prefix_metadata[..prefix_metadata_len],
                &externals[..external_count],
            )
            .map_err(not_queued)?;
            if let Some((suffix, suffix_plan)) = &suffix_parts {
                validate_output_target(
                    core,
                    suffix_plan,
                    &suffix.inputs,
                    suffix.state,
                    suffix.output,
                    &prefix_metadata[..prefix_metadata_len],
                    &externals[..external_count],
                )
                .map_err(not_queued)?;
                let prefix_range =
                    view_range(prefix_output, "prefix output").map_err(not_queued)?;
                require_disjoint_view(prefix_range, suffix.output, "suffix output")
                    .map_err(not_queued)?;
                for input in [suffix.inputs.qkv, suffix.inputs.z, suffix.inputs.ab] {
                    require_disjoint_view(prefix_range, input, "suffix input")
                        .map_err(not_queued)?;
                }
                let suffix_range =
                    view_range(suffix.output, "suffix output").map_err(not_queued)?;
                require_adjacent_outputs(prefix_range, suffix_range).map_err(not_queued)?;
                for input in [prefix_inputs.qkv, prefix_inputs.z, prefix_inputs.ab] {
                    require_disjoint_view(suffix_range, input, "prefix input")
                        .map_err(not_queued)?;
                }
            }
        }

        let owner = std::ptr::from_ref(self.core_ref().map_err(not_queued)?) as usize;
        let generation = {
            let core = self.core_mut().map_err(not_queued)?;
            core.metadata_generation = core
                .metadata_generation
                .checked_add(1)
                .ok_or_else(|| not_queued(infer("GDN block generation exhausted")))?;
            core.metadata_generation
        };

        let prefix = GdnBlockFrame {
            key: GdnBlockFrameKey {
                owner,
                metadata_generation: generation,
                slot: 0,
                tokens: prefix_plan.tokens,
                sequences: prefix_plan.sequences,
                cached: false,
                cumulative_len: prefix_cumulative.len(),
                history_cu_len: 0,
                sources_len: 0,
                write_final_state: true,
            },
            inputs: prefix_inputs,
            cumulative: prefix_cumulative,
            state: GdnStateViews::Fresh,
            metadata: GdnBlockFrameMetadata {
                cumulative: metadata.prefix_cu,
                history_cu: None,
                sources: None,
            },
            final_state: Some(prefix_final_state),
            output: prefix_output,
            write_final_state: true,
        };
        let suffix = match suffix_parts {
            None => None,
            Some((suffix, suffix_plan)) => {
                let (history_cu_len, sources_len) = match suffix.state {
                    GdnStateViews::Cached {
                        history_cumulative,
                        sources,
                        ..
                    } => (history_cumulative.len(), sources.len()),
                    GdnStateViews::Fresh => {
                        return Err(not_queued(infer(
                            "GDN block suffix requires the cached state form",
                        )))
                    }
                };
                Some(GdnBlockFrame {
                    key: GdnBlockFrameKey {
                        owner,
                        metadata_generation: generation,
                        slot: 1,
                        tokens: suffix_plan.tokens,
                        sequences: suffix_plan.sequences,
                        cached: true,
                        cumulative_len: suffix.cumulative.len(),
                        history_cu_len,
                        sources_len,
                        write_final_state: suffix.final_state.is_some(),
                    },
                    inputs: suffix.inputs,
                    cumulative: suffix.cumulative,
                    state: suffix.state,
                    metadata: GdnBlockFrameMetadata {
                        cumulative: metadata.suffix_cu,
                        history_cu: Some(metadata.history_cu),
                        sources: Some(metadata.sources),
                    },
                    final_state: suffix.final_state,
                    output: suffix.output,
                    write_final_state: suffix.final_state.is_some(),
                })
            }
        };
        Ok(GdnBlockPair { prefix, suffix })
    }

    /// Enqueue exactly one prepared block frame on the already bound stream.
    ///
    /// # Safety
    /// The outer Work retains this Prepared, the stream, every borrowed weight,
    /// input/history/slab allocation, all block metadata, external state and
    /// combined output targets through outer completion. No per-layer
    /// completion or Drop occurs. Same-stream phase reuse is allowed only after
    /// every earlier consumer was queued. On any error the caller must drain or
    /// quarantine the whole stream.
    #[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 B1 block integration.
    pub(crate) unsafe fn enqueue_in_block_frame<'f>(
        &mut self,
        frame: GdnBlockFrame<'f>,
        layer_weights: &GdnDeviceWeights,
    ) -> Result<GdnBlockOutput<'f>, GdnOwnedFailure> {
        let core = self.core_ref().map_err(not_queued)?;
        let (cached, history_cu_len, sources_len) = match frame.state {
            GdnStateViews::Fresh => (false, 0, 0),
            GdnStateViews::Cached {
                history_cumulative,
                sources,
                ..
            } => (true, history_cumulative.len(), sources.len()),
        };
        validate_block_frame_key(
            &frame.key,
            std::ptr::from_ref(core) as usize,
            core.metadata_generation,
            frame.cumulative,
            cached,
            history_cu_len,
            sources_len,
            frame.write_final_state,
        )
        .map_err(not_queued)?;
        layer_weights
            .validate_binding(self.geometry, self.stream.context())
            .map_err(not_queued)?;
        let plan = self
            .preflight_views(
                &frame.inputs,
                frame.cumulative,
                frame.state,
                frame.write_final_state,
            )
            .map_err(not_queued)?;
        if plan.tokens != frame.key.tokens || plan.sequences != frame.key.sequences {
            return Err(not_queued(infer(
                "GDN block frame request geometry changed",
            )));
        }
        validate_metadata_view(
            core,
            frame.metadata.cumulative,
            frame.cumulative.len(),
            4,
            "frame cumulative metadata",
        )
        .map_err(not_queued)?;
        let metadata_views = match frame.state {
            GdnStateViews::Fresh => [
                frame.metadata.cumulative,
                frame.metadata.cumulative,
                frame.metadata.cumulative,
            ],
            GdnStateViews::Cached {
                history_cumulative,
                sources,
                ..
            } => {
                let history_cu = frame
                    .metadata
                    .history_cu
                    .ok_or_else(|| not_queued(infer("GDN block history metadata is missing")))?;
                let source_view = frame
                    .metadata
                    .sources
                    .ok_or_else(|| not_queued(infer("GDN block source metadata is missing")))?;
                validate_metadata_view(
                    core,
                    history_cu,
                    history_cumulative.len(),
                    4,
                    "frame history cumulative metadata",
                )
                .map_err(not_queued)?;
                validate_source_view(core, source_view, sources.len(), "frame source map")
                    .map_err(not_queued)?;
                [frame.metadata.cumulative, history_cu, source_view]
            }
        };
        if let Some(target) = frame.final_state {
            validate_final_target(
                core,
                &frame.inputs,
                frame.state,
                target,
                &plan,
                &metadata_views,
            )
            .map_err(not_queued)?;
        } else if plan.final_elements.is_some() {
            return Err(not_queued(infer(
                "GDN block final-state write has no external target",
            )));
        }
        validate_output_target(
            core,
            &plan,
            &frame.inputs,
            frame.state,
            frame.output,
            &metadata_views,
            frame.final_state.as_slice(),
        )
        .map_err(not_queued)?;
        let bindings = GdnEnqueueBindings {
            cumulative: frame.metadata.cumulative.cu_deviceptr(),
            history_cu: frame
                .metadata
                .history_cu
                .map_or(0, |view| view.cu_deviceptr()),
            sources: frame.metadata.sources.map_or(0, |view| view.cu_deviceptr()),
            final_state: frame.final_state.map_or(0, |view| view.cu_deviceptr()),
            output: frame.output.cu_deviceptr(),
        };
        let mut queued = false;
        enqueue_body_with_bindings(
            core,
            &plan,
            &frame.inputs,
            layer_weights,
            bindings,
            &mut queued,
        )
        .map_err(|failure| GdnOwnedFailure {
            error: failure.error,
            queued: failure.queued,
        })?;
        Ok(GdnBlockOutput {
            output: frame.output,
            final_state: frame.final_state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(slot: u8, cached: bool, generation: u64) -> GdnBlockFrameKey {
        GdnBlockFrameKey {
            owner: 0x4000,
            metadata_generation: generation,
            slot,
            tokens: 7,
            sequences: 1,
            cached,
            cumulative_len: 2,
            history_cu_len: if cached { 2 } else { 0 },
            sources_len: if cached { 1 } else { 0 },
            write_final_state: true,
        }
    }

    #[test]
    fn byte_ranges_reject_overlap_and_allow_adjacent_targets() {
        assert!(require_disjoint((10, 20), (19, 30), "target").is_err());
        assert!(require_disjoint((10, 20), (15, 16), "target").is_err());
        assert!(require_disjoint((10, 20), (20, 30), "target").is_ok());
        assert!(require_disjoint((10, 20), (0, 10), "target").is_ok());
    }

    #[test]
    fn combined_output_requires_ordered_adjacency() {
        assert!(require_adjacent_outputs((10, 20), (20, 30)).is_ok());
        assert!(require_adjacent_outputs((10, 20), (21, 30)).is_err());
        assert!(require_adjacent_outputs((20, 30), (10, 20)).is_err());
        assert!(require_adjacent_outputs((10, 21), (20, 30)).is_err());
    }

    #[test]
    fn block_frames_accept_both_slots_at_one_generation() {
        let cumulative = [0u32, 7];
        assert!(validate_block_frame_key(
            &key(0, false, 9),
            0x4000,
            9,
            &cumulative,
            false,
            0,
            0,
            true
        )
        .is_ok());
        assert!(validate_block_frame_key(
            &key(1, true, 9),
            0x4000,
            9,
            &cumulative,
            true,
            2,
            1,
            true
        )
        .is_ok());
    }

    #[test]
    fn block_frames_reject_wrong_owner_slot_and_stale_generation() {
        let cumulative = [0u32, 7];
        assert!(validate_block_frame_key(
            &key(0, false, 9),
            0x5000,
            9,
            &cumulative,
            false,
            0,
            0,
            true
        )
        .is_err());
        assert!(validate_block_frame_key(
            &key(2, false, 9),
            0x4000,
            9,
            &cumulative,
            false,
            0,
            0,
            true
        )
        .is_err());
        assert!(validate_block_frame_key(
            &key(1, false, 9),
            0x4000,
            9,
            &cumulative,
            false,
            0,
            0,
            true
        )
        .is_err());
        assert!(validate_block_frame_key(
            &key(0, false, 9),
            0x4000,
            10,
            &cumulative,
            false,
            0,
            0,
            true
        )
        .is_err());
    }
}
