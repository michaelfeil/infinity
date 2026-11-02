// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Generation-bound owned frames and cold scratch reservation for the prepared
//! Qwen3.5 Gated DeltaNet operator. The parent module owns the exact kernel
//! pipeline and metadata staging; this module only binds one staged generation
//! to later per-layer body enqueues.

use super::*;
use std::mem::size_of;

/// Exact requested logical scratch bytes, not an NVML resident measurement.
/// Weights, modules, runtime/driver detail, and Vec control structs are excluded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnScratchBytes {
    pub device_bytes: usize,
    pub host_metadata_bytes: usize,
}

fn add_bytes(total: &mut usize, elements: usize, width: usize) -> Result<(), BackendError> {
    let part = elements
        .checked_mul(width)
        .ok_or_else(|| start("GDN scratch byte extent overflows"))?;
    *total = total
        .checked_add(part)
        .ok_or_else(|| start("GDN scratch byte total overflows"))?;
    Ok(())
}

fn aot_workspace_bytes(major: i32, sm_count: i32) -> Result<usize, BackendError> {
    let factor = match major {
        9 | 12 => 128usize,
        10 => 512usize,
        _ => {
            return Err(start(format!(
                "SM{major} has no default-dense GDN AOT workspace"
            )))
        }
    };
    if sm_count <= 0 {
        return Err(start("AOT SM count must be positive"));
    }
    usize::try_from(sm_count)
        .ok()
        .and_then(|count| count.checked_mul(factor))
        .ok_or_else(|| start("AOT workspace byte extent overflows"))
}

impl GdnPrepared {
    /// Report the exact requested reusable scratch bytes for the default dense
    /// AOT pipeline. The calculation allocates nothing and issues no CUDA query.
    pub fn scratch_bytes_default_dense(
        geometry: GdnGeometry,
        capacity: GdnCapacity,
        major: i32,
        sm_count: i32,
    ) -> Result<GdnScratchBytes, BackendError> {
        let scratch = GdnScratchLayout::new(geometry, capacity)?;
        let aot = GdnAotLayout::new(geometry, capacity)?;
        let workspace = aot_workspace_bytes(major, sm_count)?;
        let mut device_bytes = 0usize;
        add_bytes(&mut device_bytes, scratch.convolution, size_of::<bf16>())?;
        add_bytes(&mut device_bytes, scratch.qk, size_of::<f32>())?;
        add_bytes(&mut device_bytes, scratch.decay_beta, size_of::<f32>())?;
        add_bytes(&mut device_bytes, scratch.recurrence, size_of::<bf16>())?;
        add_bytes(&mut device_bytes, scratch.output, size_of::<bf16>())?;
        add_bytes(&mut device_bytes, scratch.final_state, size_of::<f32>())?;
        add_bytes(&mut device_bytes, scratch.cumulative, size_of::<u32>())?;
        add_bytes(&mut device_bytes, scratch.sources, size_of::<i64>())?;
        add_bytes(
            &mut device_bytes,
            scratch.history_cumulative,
            size_of::<i32>(),
        )?;
        add_bytes(&mut device_bytes, aot.packed_qk, size_of::<bf16>())?;
        add_bytes(&mut device_bytes, aot.packed_qk, size_of::<bf16>())?;
        add_bytes(&mut device_bytes, aot.packed_v, size_of::<bf16>())?;
        add_bytes(&mut device_bytes, aot.gates, size_of::<f32>())?;
        add_bytes(&mut device_bytes, aot.gates, size_of::<f32>())?;
        add_bytes(&mut device_bytes, aot.state, size_of::<f32>())?;
        add_bytes(&mut device_bytes, aot.state, size_of::<f32>())?;
        add_bytes(&mut device_bytes, workspace, 1)?;
        let mut host_metadata_bytes = 0usize;
        add_bytes(
            &mut host_metadata_bytes,
            scratch.cumulative,
            size_of::<u32>(),
        )?;
        add_bytes(&mut host_metadata_bytes, scratch.sources, size_of::<i64>())?;
        add_bytes(
            &mut host_metadata_bytes,
            scratch.history_cumulative,
            size_of::<i32>(),
        )?;
        Ok(GdnScratchBytes {
            device_bytes,
            host_metadata_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GdnFrameKey {
    owner: usize,
    metadata_generation: u64,
    tokens: i32,
    sequences: i32,
    cached: bool,
    cumulative_len: usize,
    history_cu_len: usize,
    sources_len: usize,
}

impl GdnFrameKey {
    fn capture(
        core: &GdnCore,
        plan: &GdnSubmitPlan,
        cumulative_len: usize,
        state: &GdnState<'_>,
    ) -> Self {
        let (cached, history_cu_len, sources_len) = match state {
            GdnState::Fresh => (false, 0, 0),
            GdnState::Cached(cached) => (
                true,
                cached.history.cumulative.len(),
                cached.history.sources.len(),
            ),
        };
        Self {
            owner: std::ptr::from_ref(core) as usize,
            metadata_generation: core.metadata_generation,
            tokens: plan.tokens,
            sequences: plan.sequences,
            cached,
            cumulative_len,
            history_cu_len,
            sources_len,
        }
    }
}

/// Opaque copy token binding one staged metadata generation to prepared owner.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 engine integration.
pub(crate) struct GdnOwnedFrame {
    key: GdnFrameKey,
    cumulative: u64,
}

fn not_queued(error: BackendError) -> GdnOwnedFailure {
    GdnOwnedFailure {
        error,
        queued: false,
    }
}

fn queued_failure(error: BackendError) -> GdnOwnedFailure {
    GdnOwnedFailure {
        error,
        queued: true,
    }
}

fn uploaded_cumulative_binding_bytes(
    cumulative_len: usize,
    uploaded_len: usize,
    uploaded_alignment: usize,
) -> Result<usize, BackendError> {
    if uploaded_len != cumulative_len {
        return Err(infer(format!(
            "uploaded cumulative metadata has {uploaded_len} entries, expected {cumulative_len}"
        )));
    }
    if uploaded_alignment < 4 || !uploaded_alignment.is_multiple_of(4) {
        return Err(infer(
            "uploaded cumulative metadata base alignment is below four bytes",
        ));
    }
    metadata_bytes::<u32>(cumulative_len)
}

fn validate_frame_key(
    key: GdnFrameKey,
    owner: usize,
    generation: u64,
    staged_cu: &[u32],
    staged_history_cu: &[i32],
    staged_sources: &[i64],
    cached_metadata: Option<(&[i32], &[i64])>,
) -> Result<(), BackendError> {
    if key.owner != owner {
        return Err(infer(
            "GDN frame token belongs to a different prepared owner",
        ));
    }
    if key.metadata_generation != generation {
        return Err(infer("GDN frame token metadata generation is stale"));
    }
    if staged_cu.len() != key.cumulative_len {
        return Err(infer("GDN frame cumulative metadata changed"));
    }
    let (history_cu, sources) = match (key.cached, cached_metadata) {
        (false, None) => return Ok(()),
        (true, Some(metadata)) => metadata,
        _ => return Err(infer("GDN frame token cached mode changed")),
    };
    if history_cu.len() != key.history_cu_len || sources.len() != key.sources_len {
        return Err(infer("GDN frame cached history metadata length changed"));
    }
    if staged_history_cu.len() != key.history_cu_len || staged_sources.len() != key.sources_len {
        return Err(infer("GDN frame cached history metadata changed"));
    }
    let (Some(staged_history), Some(staged_source_map)) = (
        staged_history_cu.get(..key.history_cu_len),
        staged_sources.get(..key.sources_len),
    ) else {
        return Err(infer("GDN frame cached history metadata is unavailable"));
    };
    if history_cu != staged_history || sources != staged_source_map {
        return Err(infer("GDN frame cached history metadata changed"));
    }
    Ok(())
}

impl GdnPrepared {
    /// Admit one frame, stage its host metadata once, and upload it exactly once.
    ///
    /// # Safety
    /// The outer Work owns this Prepared, stream, input/history allocations, and
    /// immutable CPU metadata through completion, with this context bound. On ANY
    /// error it must drain or quarantine the stream including earlier layers.
    #[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 engine integration.
    pub(crate) unsafe fn prepare_owned_frame(
        &mut self,
        inputs: GdnInputViews<'_>,
        cumulative: &[u32],
        state: GdnState<'_>,
        write_final_state: bool,
    ) -> Result<GdnOwnedFrame, GdnOwnedFailure> {
        let plan = self
            .preflight(&inputs, cumulative, &state, write_final_state)
            .map_err(not_queued)?;
        let core = self.core_mut().map_err(not_queued)?;
        stage_host_metadata(core, cumulative, &state).map_err(not_queued)?;
        let key = GdnFrameKey::capture(core, &plan, cumulative.len(), &state);
        let mut queued = false;
        enqueue_metadata(core, &plan, &mut queued).map_err(|failure| GdnOwnedFailure {
            error: failure.error,
            queued: failure.queued,
        })?;
        Ok(GdnOwnedFrame {
            key,
            cumulative: core.cu.cu_deviceptr(),
        })
    }

    /// Admit one fresh frame whose cumulative metadata is already device-resident.
    ///
    /// The engine path reuses the Work's existing uploaded cumulative view and
    /// binds that checked device range directly without any metadata copy; it
    /// never re-uploads host metadata. Borrowing that already-uploaded input
    /// avoids the per-request `core.host_cu` H2D race that lets another engine
    /// mutate the shared host vector before an earlier DMA completes. Layer
    /// bodies still add no metadata transfer, and a future prefix second slot is
    /// explicitly outside this change.
    ///
    /// # Safety
    /// The outer Work owns this Prepared, stream, every input/weight binding,
    /// and both the device allocation backing `uploaded_cumulative` and its
    /// producer until the enclosing stream/Work completion. `uploaded_cumulative` must hold
    /// exactly `cumulative.len()` valid `u32` words produced by an already
    /// enqueued upload on this stream, or be immutable Work input. Same-stream
    /// scratch/input reuse is allowed only after the previous use is queued; no
    /// owner may be CPU-mutated or reused while a HostIoLease pins it. On ANY
    /// error the Work must drain or quarantine the stream including earlier
    /// layers. Any earlier standalone or owned-host-metadata DMA must already
    /// be complete before calling this method; safe GdnSubmission borrows enforce
    /// that for the public path. Engine Works exclusively use this direct path.
    #[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 engine integration.
    pub(crate) unsafe fn prepare_owned_fresh_frame_from_device(
        &mut self,
        inputs: GdnInputViews<'_>,
        cumulative: &[u32],
        uploaded_cumulative: GdnBufferView<u32>,
    ) -> Result<GdnOwnedFrame, GdnOwnedFailure> {
        let state = GdnState::Fresh;
        let plan = self
            .preflight(&inputs, cumulative, &state, false)
            .map_err(not_queued)?;
        let binding_bytes = uploaded_cumulative_binding_bytes(
            cumulative.len(),
            uploaded_cumulative.len(),
            pointer_alignment(uploaded_cumulative.cu_deviceptr()),
        )
        .map_err(not_queued)?;
        if binding_bytes != plan.cu_bytes {
            return Err(not_queued(infer(
                "uploaded cumulative metadata byte extent differs from the plan",
            )));
        }
        let core = self.core_mut().map_err(not_queued)?;
        require_context(
            &core.stream,
            uploaded_cumulative.context(),
            "uploaded cumulative metadata",
        )
        .map_err(not_queued)?;
        stage_host_metadata(core, cumulative, &state).map_err(not_queued)?;
        let key = GdnFrameKey::capture(core, &plan, cumulative.len(), &state);
        let cumulative = uploaded_cumulative.cu_deviceptr();
        if let Some(work) = core.aot.as_ref() {
            checked_aot_cumulative(core, work, cumulative).map_err(not_queued)?;
        }
        // No GPU transfer is necessary: the common HostIoLease upload already
        // placed these exact u32/int32-compatible offsets on this stream.
        // core.host_cu is only CPU preflight metadata on this path, never a DMA source.
        Ok(GdnOwnedFrame { key, cumulative })
    }

    /// Enqueue one owned layer body against an already prepared frame.
    ///
    /// # Safety
    /// The outer Work owns this Prepared, stream, EVERY borrowed per-layer weight
    /// binding, input/history allocation, and the frame's CPU metadata until
    /// completion. Ordered same-stream slab overwrite is allowed; CPU or
    /// other-stream mutation/reuse/eviction is not. On ANY error it must drain or
    /// quarantine the stream including earlier layers and borrowed weights.
    #[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 engine integration.
    pub(crate) unsafe fn enqueue_owned_in_frame<'a>(
        &'a mut self,
        frame: &GdnOwnedFrame,
        inputs: GdnInputViews<'_>,
        state: GdnState<'_>,
        layer_weights: &GdnDeviceWeights,
        write_final_state: bool,
    ) -> Result<GdnOwnedOutput<'a>, GdnOwnedFailure> {
        let key = frame.key;
        let core = self.core_ref().map_err(not_queued)?;
        let cached_metadata = match &state {
            GdnState::Fresh => None,
            GdnState::Cached(cached) => Some((cached.history.cumulative, cached.history.sources)),
        };
        validate_frame_key(
            key,
            std::ptr::from_ref(core) as usize,
            core.metadata_generation,
            &core.host_cu,
            &core.host_history_cu,
            &core.host_sources,
            cached_metadata,
        )
        .map_err(not_queued)?;
        let cumulative = core
            .host_cu
            .get(..key.cumulative_len)
            .ok_or_else(|| not_queued(infer("GDN frame cumulative metadata is unavailable")))?;
        layer_weights
            .validate_binding(self.geometry, self.stream.context())
            .map_err(not_queued)?;
        let plan = self
            .preflight(&inputs, cumulative, &state, write_final_state)
            .map_err(not_queued)?;
        if plan.tokens != key.tokens || plan.sequences != key.sequences {
            return Err(not_queued(infer(
                "GDN frame token request geometry changed",
            )));
        }
        let mut queued = false;
        enqueue_body(
            core,
            &plan,
            &inputs,
            layer_weights,
            frame.cumulative,
            &mut queued,
        )
        .map_err(|failure| GdnOwnedFailure {
            error: failure.error,
            queued: failure.queued,
        })?;
        let output = GdnBufferView::from_buffer_range(&core.output, 0..plan.output_elements)
            .map_err(queued_failure)?;
        let final_state = match plan.final_elements {
            Some(elements) => Some(
                GdnBufferView::from_buffer_range(&core.final_state, 0..elements)
                    .map_err(queued_failure)?,
            ),
            None => None,
        };
        Ok(GdnOwnedOutput {
            output,
            final_state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tokens: usize, value_heads: usize, major: i32) -> GdnScratchBytes {
        GdnPrepared::scratch_bytes_default_dense(
            GdnGeometry::new(16, value_heads, 4).expect("geometry"),
            GdnCapacity::new(tokens, 1, 1).expect("capacity"),
            major,
            132,
        )
        .expect("scratch bytes")
    }

    fn frame_key(cached: bool, history_cu_len: usize, sources_len: usize, gen: u64) -> GdnFrameKey {
        GdnFrameKey {
            owner: 0x1000,
            metadata_generation: gen,
            tokens: 7,
            sequences: 2,
            cached,
            cumulative_len: 3,
            history_cu_len,
            sources_len,
        }
    }

    #[derive(Clone, Copy)]
    struct Probe<'a> {
        owner: usize,
        generation: u64,
        cumulative: &'a [u32],
    }

    macro_rules! check {
        ($key:expr, $probe:expr, $staged:expr, $provided:expr) => {
            validate_frame_key(
                $key,
                ($probe).owner,
                ($probe).generation,
                ($probe).cumulative,
                ($staged).0,
                ($staged).1,
                ($key).cached.then_some($provided),
            )
            .is_ok()
        };
    }

    #[test]
    fn scratch_bytes_match_independent_default_dense_totals() {
        for (tokens, value_heads, major, device_bytes) in [
            (128, 48, 9, 20_038_168),
            (2048, 48, 9, 178_799_128),
            (128, 32, 9, 14_762_520),
            (2048, 32, 9, 141_574_680),
        ] {
            assert_eq!(
                scratch(tokens, value_heads, major),
                GdnScratchBytes {
                    device_bytes,
                    host_metadata_bytes: 24,
                }
            );
        }
        assert_eq!(
            scratch(128, 48, 10).device_bytes - scratch(128, 48, 9).device_bytes,
            50_688
        );
    }

    #[test]
    fn scratch_bytes_reject_overflow_unsupported_sm_and_bad_counts() {
        let geometry = GdnGeometry::new(16, 48, 4).expect("geometry");
        let capacity = GdnCapacity::new(128, 1, 1).expect("capacity");
        for (major, sm_count) in [(8, 132), (11, 132), (9, 0), (9, -1)] {
            assert!(
                GdnPrepared::scratch_bytes_default_dense(geometry, capacity, major, sm_count)
                    .is_err()
            );
        }
        assert!(GdnCapacity::new(0, 0, 0).is_err());
        assert!(GdnCapacity::new(1, 2, 0).is_err());
        let too_many = GdnCapacity::new(GDN_MAX_TOKENS, 1, 0).expect("capacity");
        let wide = GdnGeometry::new(1, 2, 1).expect("geometry");
        assert!(GdnPrepared::scratch_bytes_default_dense(wide, too_many, 9, 132).is_err());
        let mut total = 0usize;
        assert!(add_bytes(&mut total, usize::MAX, 2).is_err());
        let mut overflowed = usize::MAX;
        assert!(add_bytes(&mut overflowed, 1, 1).is_err());
    }

    #[test]
    fn frame_key_rejects_identity_metadata_and_cached_length_changes() {
        let cumulative = [0u32, 3, 7];
        let empty = (&[][..], &[][..]);
        let fresh = frame_key(false, 0, 0, 4);
        let mut probe = Probe {
            owner: 0x1000,
            generation: 4,
            cumulative: &cumulative,
        };
        assert!(!check!(
            fresh,
            Probe {
                owner: 0x2000,
                ..probe
            },
            empty,
            empty
        ));
        probe.generation = 5;
        assert!(!check!(fresh, probe, empty, empty));
        probe.generation = 4;
        assert!(check!(fresh, probe, empty, empty));

        let history = [0i32, 1, 2];
        let sources = [0i64, 0];
        let staged = (&history[..], &sources[..]);
        let cached = frame_key(true, history.len(), sources.len(), 3);
        let cumulative = [0u32, 2, 5];
        probe = Probe {
            generation: 3,
            cumulative: &cumulative,
            ..probe
        };
        assert!(check!(cached, probe, staged, staged));
        let changed_history = [0i32, 1, 3];
        let changed_sources = [0i64, 1];
        let changed_history = (&changed_history[..], &sources[..]);
        let changed_sources = (&history[..], &changed_sources[..]);
        assert!(!check!(cached, probe, staged, changed_history));
        assert!(!check!(cached, probe, staged, changed_sources));
        assert!(validate_frame_key(
            cached,
            probe.owner,
            probe.generation,
            probe.cumulative,
            &history,
            &sources,
            None
        )
        .is_err());

        let geometry = GdnGeometry::new(16, 48, 4).expect("geometry");
        let capacity = GdnCapacity::new(256, 1, 0).expect("capacity");
        let plan_255 = GdnPlan::new(geometry, capacity, &[0, 255]).expect("plan");
        let plan_256 = GdnPlan::new(geometry, capacity, &[0, 256]).expect("plan");
        assert!(!aot_admits(true, plan_255.tokens(), plan_255.sequences()));
        assert!(aot_admits(true, plan_256.tokens(), plan_256.sequences()));
        let mut key_255 = frame_key(true, 0, 0, 1);
        key_255.tokens = plan_255.tokens() as i32;
        key_255.sequences = plan_255.sequences() as i32;
        let mut key_256 = key_255;
        key_256.tokens = plan_256.tokens() as i32;
        assert_ne!(key_255.tokens, key_256.tokens);
    }

    #[test]
    fn uploaded_cumulative_requires_exact_extent_four_byte_alignment_and_size() {
        assert_eq!(
            uploaded_cumulative_binding_bytes(257, 257, 4).expect("exact extent"),
            257 * size_of::<u32>()
        );
        assert!(uploaded_cumulative_binding_bytes(257, 256, 8).is_err());
        assert!(uploaded_cumulative_binding_bytes(257, 258, 8).is_err());
        assert!(uploaded_cumulative_binding_bytes(257, 257, 2).is_err());
        assert!(uploaded_cumulative_binding_bytes(257, 257, 0).is_err());
        assert!(uploaded_cumulative_binding_bytes(257, 257, 6).is_err());
        assert!(metadata_bytes::<u32>(usize::MAX).is_err());
    }
}
