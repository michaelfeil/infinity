// Copyright (c) 2026 Oxide MoE Authors. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Infinity-Restricted-MIT-1.0
//
// Original Rust source offered under the Infinity Restricted MIT License 1.0
// (see LICENSE). This is not standard MIT; Appendix A provider use is prohibited.

//! Owned GPU runtime wrapper for the native routed Qwen3 MoE ABI.
//!
//! One [`MoeFrame`] prepares a single token-exact plan and router descriptor
//! that every routed decoder layer reuses. [`MoeScratch`] owns the retained
//! per-request arena, and [`MoeProvider::forward`] enqueues exactly one native
//! routed operation per layer without per-expert launches, heap work, or a
//! context bind. Callers bind the context and retain every owner until the
//! stream drains; this module never synchronizes inside a forward call.

use crate::{
    cublas::Blas,
    lt::ShapeError,
    math::LinearPlan,
    moe_abi::{Buffers, ModelInfo, Shape, ABI_VERSION, KERNEL_AUTO, MODEL_QWEN3_E128_TOP8},
    moe_api::{Api, PreparedPlan},
    moe_model::{GateUpSources, RoutedLayerWeights, RoutedMlpConfig},
    tensor::Bf16,
    upload::WeightUploader,
};
use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::{ffi::c_void, mem::ManuallyDrop, path::Path, sync::Arc};

use super::{infer, product, start, u32_size};

const WORKSPACE_ALIGNMENT: usize = 256;
const QWEN3_EXPERTS: usize = 128;
const QWEN3_TOP_K: usize = 8;
const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;
const CAST_THREADS: u32 = 256;

fn classify_metadata_error(error: crate::moe_api::Error) -> ShapeError {
    use crate::moe_api::Error;
    match error {
        Error::Overflow
        | Error::ShapeMismatch(_)
        | Error::WorkspaceTooSmall
        | Error::InvalidArgument(_) => ShapeError::Rejected(infer(error)),
        _ => ShapeError::Fatal(infer(error)),
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, BackendError> {
    if alignment == 0 {
        return Err(infer("MoE workspace alignment must be positive"));
    }
    value
        .checked_add(alignment - 1)
        .map(|n| n / alignment * alignment)
        .ok_or_else(|| infer("MoE workspace size overflow"))
}

/// The ABI1 Qwen3 sidecar takes caller-supplied hidden/intermediate widths and
/// requires positive multiples of 8. Executor recipes still gate the actual
/// admitted architectures; this only rejects malformed structural geometry.
fn geometry_supported(hidden: usize, intermediate: usize) -> bool {
    hidden > 0 && intermediate > 0 && hidden.is_multiple_of(8) && intermediate.is_multiple_of(8)
}

fn arch_for_compute_capability(major: i32, minor: i32) -> Result<i32, BackendError> {
    let arch = major
        .checked_mul(10)
        .and_then(|n| n.checked_add(minor))
        .ok_or_else(|| start("MoE compute capability overflows the arch value"))?;
    match arch {
        90 | 100 | 120 => Ok(arch),
        other => Err(start(format!(
            "MoE provider requires sm90/sm100/sm120, found sm{other}"
        ))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScratchLayout {
    tokens: usize,
    logits: usize,
    workspace: usize,
    bytes: usize,
}

fn scratch_layout(
    tokens: usize,
    experts: usize,
    workspace_bytes: usize,
) -> Result<ScratchLayout, BackendError> {
    if tokens == 0 {
        return Err(infer("MoE scratch capacity must be positive"));
    }
    let logits = product(tokens, experts)?;
    let workspace = align_up(workspace_bytes, WORKSPACE_ALIGNMENT)?;
    let bf16 = product(logits, BF16_BYTES)?;
    let f32 = product(logits, F32_BYTES)?;
    let bytes = bf16
        .checked_add(f32)
        .and_then(|n| n.checked_add(workspace))
        .ok_or_else(|| infer("MoE scratch byte count overflow"))?;
    Ok(ScratchLayout {
        tokens,
        logits,
        workspace,
        bytes,
    })
}

fn prefix_elements(
    tokens: usize,
    capacity: usize,
    hidden: usize,
    experts: usize,
    logits_capacity: usize,
) -> Result<(usize, usize), BackendError> {
    if tokens == 0 || tokens > capacity {
        return Err(infer(
            "MoE frame tokens exceed the retained scratch capacity",
        ));
    }
    let hidden_elements = product(tokens, hidden)?;
    let logits = product(tokens, experts)?;
    if logits > logits_capacity {
        return Err(infer(
            "MoE router logits exceed the retained scratch capacity",
        ));
    }
    Ok((hidden_elements, logits))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LayerElements {
    router: usize,
    gate_up: usize,
    down: usize,
}

fn layer_elements(
    experts: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<LayerElements, BackendError> {
    let two_i = product(intermediate, 2)?;
    let down_rows = product(experts, hidden)?;
    Ok(LayerElements {
        router: down_rows,
        gate_up: product(product(experts, two_i)?, hidden)?,
        down: product(down_rows, intermediate)?,
    })
}

fn same_context(actual: &Arc<Context>, expected: &Arc<Context>) -> bool {
    Arc::ptr_eq(actual, expected)
}

/// Canonical BF16 expert weights for one routed layer.
pub(crate) struct MoeLayer {
    router: DeviceBuffer<Bf16>,
    gate_up: DeviceBuffer<Bf16>,
    down: DeviceBuffer<Bf16>,
    hidden: usize,
    intermediate: usize,
    experts: usize,
}

impl MoeLayer {
    /// Upload the canonical router `[E,H]`, gate/up `[E,2I,H]` and down
    /// `[E,H,I]` tensors once, directly into their final layouts. Separate
    /// expert views are staged cold in `[e gate, e up]` order; no transpose or
    /// duplicate device copy runs here.
    pub(crate) fn upload(
        uploader: &mut WeightUploader,
        host: &RoutedLayerWeights,
        config: &RoutedMlpConfig,
    ) -> Result<Self, BackendError> {
        let hidden = config.hidden_size;
        let intermediate = config.moe_intermediate_size;
        let experts = config.num_experts;
        if hidden == 0 || intermediate == 0 || experts == 0 {
            return Err(start("routed layer geometry must be positive"));
        }
        let two_i = product(intermediate, 2)?;
        let canonical_sources = product(experts, 2)?;
        let router_shape = [experts, hidden];
        if host.router.shape() != router_shape.as_slice() {
            return Err(start("routed router weight shape differs from config"));
        }
        let packed_shape = [two_i, hidden];
        let separate_shape = [intermediate, hidden];
        let down_shape = [hidden, intermediate];

        let mut views = Vec::new();
        views
            .try_reserve_exact(experts)
            .map_err(|_| start("routed expert view allocation failed"))?;
        for index in 0..experts {
            views.push(host.expert(index)?);
        }
        let mut gate_sources = Vec::new();
        gate_sources
            .try_reserve_exact(canonical_sources)
            .map_err(|_| start("routed gate/up view allocation failed"))?;
        let mut down_sources = Vec::new();
        down_sources
            .try_reserve_exact(experts)
            .map_err(|_| start("routed down view allocation failed"))?;
        let mut fused = None;
        for view in &views {
            match &view.gate_up {
                GateUpSources::Packed(source) => {
                    if source.shape() != packed_shape.as_slice() || fused == Some(false) {
                        return Err(start("fused gate/up expert shape differs from config"));
                    }
                    fused = Some(true);
                    gate_sources.push(source);
                }
                GateUpSources::Separate { gate, up } => {
                    if gate.shape() != separate_shape.as_slice()
                        || up.shape() != separate_shape.as_slice()
                        || fused == Some(true)
                    {
                        return Err(start("separate gate/up expert shape differs from config"));
                    }
                    fused = Some(false);
                    gate_sources.push(gate);
                    gate_sources.push(up);
                }
            }
            if view.down.shape() != down_shape.as_slice() {
                return Err(start("routed down expert shape differs from config"));
            }
            down_sources.push(&view.down);
        }
        let expected_sources = if fused == Some(true) {
            experts
        } else {
            canonical_sources
        };
        if gate_sources.len() != expected_sources {
            return Err(start("routed gate/up source count differs from config"));
        }

        let router = uploader.upload(&[&host.router])?;
        let gate_up = uploader.upload(&gate_sources)?;
        let down = uploader.upload(&down_sources)?;
        Ok(Self {
            router,
            gate_up,
            down,
            hidden,
            intermediate,
            experts,
        })
    }
}

/// A cold-loaded, validated native ABI provider with its geometry fixed.
pub(crate) struct MoeProvider {
    api: Arc<Api>,
    context: Arc<Context>,
    _model: ModelInfo,
    hidden: usize,
    intermediate: usize,
    experts: usize,
    top_k: usize,
    renormalize: i32,
}

impl MoeProvider {
    /// Load the trusted native ABI1 artifact for the context's architecture.
    ///
    /// # Safety
    ///
    /// `path` must name trusted native code compiled for the matching
    /// `oxide_moe.h` ABI. The owner resolves it from the pinned kernel bundle;
    /// loading runs platform initializers before any validation.
    pub(crate) unsafe fn load(
        context: &Arc<Context>,
        path: &Path,
        config: &RoutedMlpConfig,
    ) -> Result<Self, BackendError> {
        if config.num_experts != QWEN3_EXPERTS || config.num_experts_per_tok != QWEN3_TOP_K {
            return Err(start("Qwen3 routed MoE requires 128 experts and top_k 8"));
        }
        if !geometry_supported(config.hidden_size, config.moe_intermediate_size) {
            return Err(start(
                "Qwen3 routed MoE geometry must have positive hidden/intermediate multiples of 8",
            ));
        }
        let (major, minor) = context.compute_capability().map_err(start)?;
        let arch = arch_for_compute_capability(major, minor)?;
        // SAFETY: the caller promises a trusted artifact resolved by the owner.
        let api = unsafe { Api::load(path, arch) }.map_err(start)?;
        let model = api.query_model(MODEL_QWEN3_E128_TOP8).map_err(start)?;
        if model.experts != QWEN3_EXPERTS as i32 || model.top_k != QWEN3_TOP_K as i32 {
            return Err(start(
                "native ABI1 Qwen3 model geometry differs from config",
            ));
        }
        Ok(Self {
            api,
            context: Arc::clone(context),
            _model: model,
            hidden: config.hidden_size,
            intermediate: config.moe_intermediate_size,
            experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            renormalize: if config.norm_topk_prob { 1 } else { 0 },
        })
    }

    fn shape(&self, tokens: usize) -> Result<Shape, BackendError> {
        if tokens == 0 {
            return Err(infer("MoE frame token count must be positive"));
        }
        let tokens = i32::try_from(tokens).map_err(infer)?;
        let hidden = i32::try_from(self.hidden).map_err(infer)?;
        let intermediate = i32::try_from(self.intermediate).map_err(infer)?;
        let experts = i32::try_from(self.experts).map_err(infer)?;
        let top_k = i32::try_from(self.top_k).map_err(infer)?;
        Ok(Shape::new(
            MODEL_QWEN3_E128_TOP8,
            KERNEL_AUTO,
            tokens,
            hidden,
            intermediate,
            experts,
            top_k,
            self.renormalize,
        ))
    }

    /// Prepare one token-exact native plan plus the shared router descriptor.
    /// No descriptor, allocation, or stream operation is performed.
    pub(crate) fn prepare(&self, tokens: usize) -> Result<MoeFrame, ShapeError> {
        let shape = self.shape(tokens).map_err(ShapeError::Rejected)?;
        let prepared = self.api.prepare(&shape).map_err(classify_metadata_error)?;
        let router = LinearPlan::<Bf16>::contiguous(tokens, self.hidden, self.experts)
            .map_err(|error| ShapeError::Rejected(infer(error)))?;
        Ok(MoeFrame {
            provider: Arc::as_ptr(&self.api) as usize,
            prepared,
            router,
            tokens,
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.experts,
            top_k: self.top_k,
        })
    }

    fn workspace_bytes_typed(&self, tokens: usize) -> Result<usize, ShapeError> {
        let shape = self.shape(tokens).map_err(ShapeError::Rejected)?;
        self.api
            .query_workspace(&shape)
            .map(|info| info.workspace_bytes)
            .map_err(classify_metadata_error)
    }

    fn workspace_bytes(&self, tokens: usize) -> Result<usize, BackendError> {
        let shape = self.shape(tokens)?;
        self.api
            .query_workspace(&shape)
            .map(|info| info.workspace_bytes)
            .map_err(start)
    }

    /// Enqueue one native routed operation for one layer.
    ///
    /// # Safety
    ///
    /// The caller must bind the provider's context on this thread and keep the
    /// frame, layer, `blas`, kernels, stream, every device allocation and this
    /// scratch alive without conflicting access until `stream` drains. Input and
    /// output may be capacity-sized arena buffers; only the frame's token prefix
    /// is addressed. No synchronize or fallback runs here; an error is fatal to
    /// the outer owner, which drains the stream.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn forward(
        &self,
        frame: &MoeFrame,
        layer: &MoeLayer,
        blas: &Blas,
        bf16_to_f32: &Function,
        stream: &Stream,
        input: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
        scratch: &mut MoeScratch,
    ) -> Result<(), BackendError> {
        if frame.provider != Arc::as_ptr(&self.api) as usize
            || frame.hidden != self.hidden
            || frame.intermediate != self.intermediate
            || frame.experts != self.experts
            || frame.top_k != self.top_k
        {
            return Err(infer(
                "MoE frame belongs to a different provider or geometry",
            ));
        }
        if layer.hidden != self.hidden
            || layer.intermediate != self.intermediate
            || layer.experts != self.experts
        {
            return Err(infer("MoE layer geometry differs from provider"));
        }
        if !Arc::ptr_eq(blas.context(), &self.context)
            || !Arc::ptr_eq(stream.context(), &self.context)
        {
            return Err(infer("MoE provider, cuBLAS and stream contexts differ"));
        }
        if blas.stream().cu_stream() != stream.cu_stream()
            || scratch.stream.cu_stream() != stream.cu_stream()
        {
            return Err(infer("MoE router, scratch and consumer streams differ"));
        }
        for actual in [
            input.context(),
            output.context(),
            scratch.logits_bf16.context(),
            scratch.logits_f32.context(),
            scratch.workspace_buffer.context(),
            layer.router.context(),
            layer.gate_up.context(),
            layer.down.context(),
        ] {
            if !same_context(actual, &self.context) {
                return Err(infer("MoE buffers belong to a different CUDA context"));
            }
        }

        let (hidden_elements, logits_elements) = prefix_elements(
            frame.tokens,
            scratch.tokens,
            self.hidden,
            self.experts,
            scratch.logits,
        )?;
        if input.len() < hidden_elements || output.len() < hidden_elements {
            return Err(infer(
                "MoE input/output buffer is shorter than the token prefix",
            ));
        }
        let elements = layer_elements(self.experts, self.hidden, self.intermediate)?;
        if layer.router.len() != elements.router
            || layer.gate_up.len() != elements.gate_up
            || layer.down.len() != elements.down
        {
            return Err(infer(
                "MoE layer weight lengths differ from canonical geometry",
            ));
        }
        let workspace_required = frame.prepared.workspace_bytes();
        if workspace_required == 0 || workspace_required > scratch.workspace {
            return Err(infer("MoE native workspace exceeds the retained scratch"));
        }

        // SAFETY: contexts, lengths and owners were checked above; the router
        // output is the capacity-retained BF16 logits prefix.
        unsafe {
            blas.linear(
                &frame.router,
                input,
                &layer.router,
                &mut scratch.logits_bf16,
            )
        }?;
        cast_logits(
            bf16_to_f32,
            stream,
            scratch.logits_bf16.cu_deviceptr(),
            scratch.logits_f32.cu_deviceptr(),
            logits_elements,
        )?;

        let buffers = Buffers {
            struct_size: std::mem::size_of::<Buffers>() as u32,
            abi_version: ABI_VERSION,
            logits: scratch.logits_f32.cu_deviceptr() as *const f32,
            input_bf16: input.cu_deviceptr() as *const c_void,
            gate_up_weight_bf16: layer.gate_up.cu_deviceptr() as *const c_void,
            down_weight_bf16: layer.down.cu_deviceptr() as *const c_void,
            expert_scales: std::ptr::null(),
            output_bf16: output.cu_deviceptr() as *mut c_void,
            workspace: scratch.workspace_buffer.cu_deviceptr() as *mut c_void,
            workspace_bytes: workspace_required,
            stream: stream.cu_stream().cast(),
            flags: 0,
            reserved: [0; 3],
        };
        // SAFETY: the retained plan matches this frame and every buffer, owner,
        // and stream satisfies the borrowed-forward contract documented above.
        unsafe { frame.prepared.forward_bound(&buffers) }.map_err(start)
    }
}

/// One token-exact native plan and router descriptor shared by every routed
/// decoder layer in a request. Dropping it only releases metadata/library owners.
pub(crate) struct MoeFrame {
    provider: usize,
    prepared: PreparedPlan,
    router: LinearPlan<Bf16>,
    tokens: usize,
    hidden: usize,
    intermediate: usize,
    experts: usize,
    top_k: usize,
}

impl MoeFrame {
    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }
}

/// Retained per-request MoE arena: BF16 router logits, widened F32 logits and
/// the native workspace. The workspace prefix is zeroed stream-ordered once at
/// construction; every logits element is written before the native read.
pub(crate) struct MoeScratch {
    stream: Arc<Stream>,
    tokens: usize,
    logits: usize,
    workspace: usize,
    bytes: usize,
    logits_bf16: ManuallyDrop<DeviceBuffer<Bf16>>,
    logits_f32: ManuallyDrop<DeviceBuffer<f32>>,
    workspace_buffer: ManuallyDrop<DeviceBuffer<u8>>,
    released: bool,
}

impl MoeScratch {
    pub(crate) fn allocation_bytes_typed(
        provider: &MoeProvider,
        capacity_tokens: usize,
    ) -> Result<usize, ShapeError> {
        let workspace = provider.workspace_bytes_typed(capacity_tokens)?;
        scratch_layout(capacity_tokens, provider.experts, workspace)
            .map(|layout| layout.bytes)
            .map_err(ShapeError::Rejected)
    }

    /// Exact retained capacity for `capacity_tokens`, including the native
    /// workspace queried at that capacity and 256-byte alignment padding.
    pub(crate) fn allocation_bytes(
        provider: &MoeProvider,
        capacity_tokens: usize,
    ) -> Result<usize, BackendError> {
        Ok(Self::layout(provider, capacity_tokens)?.bytes)
    }

    fn layout(
        provider: &MoeProvider,
        capacity_tokens: usize,
    ) -> Result<ScratchLayout, BackendError> {
        let workspace = provider.workspace_bytes(capacity_tokens)?;
        scratch_layout(capacity_tokens, provider.experts, workspace)
    }

    pub(crate) fn new(
        provider: &MoeProvider,
        stream: Arc<Stream>,
        capacity_tokens: usize,
    ) -> Result<Self, BackendError> {
        let layout = Self::layout(provider, capacity_tokens)?;
        // A retained arena is defined before any consumer can address it. The
        // router GEMM overwrites the token prefix; this one cold stream-ordered
        // zero defines the remaining capacity without a warm synchronization.
        let mut logits_bf16 =
            unsafe { DeviceBuffer::<Bf16>::uninitialized_async(&stream, layout.logits) }
                .map_err(start)?;
        logits_bf16.zero_async(&stream).map_err(start)?;
        // SAFETY: the widening cast writes every element before the native read.
        let logits_f32 =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, layout.logits) }
                .map_err(start)?;
        // SAFETY: workspace writes are confined to the native library on the
        // ordered stream after the initialization zero below.
        let mut workspace_buffer =
            unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, layout.workspace) }
                .map_err(start)?;
        workspace_buffer.zero_async(&stream).map_err(start)?;
        Ok(Self {
            stream,
            tokens: layout.tokens,
            logits: layout.logits,
            workspace: layout.workspace,
            bytes: layout.bytes,
            logits_bf16: ManuallyDrop::new(logits_bf16),
            logits_f32: ManuallyDrop::new(logits_f32),
            workspace_buffer: ManuallyDrop::new(workspace_buffer),
            released: false,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    /// Release buffers after the owning decoder arena proved its stream complete.
    ///
    /// # Safety
    /// Every GPU use of this scratch must have completed; the owner must use the
    /// same retained stream. Idempotent so the later destructor does not drain again.
    pub(crate) unsafe fn drop_after_drain(&mut self) {
        if self.released {
            return;
        }
        // SAFETY: completion was proved by the enclosing arena.
        unsafe {
            ManuallyDrop::drop(&mut self.logits_bf16);
            ManuallyDrop::drop(&mut self.logits_f32);
            ManuallyDrop::drop(&mut self.workspace_buffer);
        }
        self.released = true;
    }
}

impl Drop for MoeScratch {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        if self.stream.synchronize().is_ok() {
            // SAFETY: this exact stream has completed every scratch use.
            unsafe { self.drop_after_drain() };
        } else {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
        }
    }
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn cast_logits(
    function: &Function,
    stream: &Stream,
    mut source: u64,
    mut destination: u64,
    elements: usize,
) -> Result<(), BackendError> {
    let mut selected: u64 = 0;
    let mut count = u32_size(elements)?;
    let mut width = 1u32;
    let grid = (u32_size(elements.div_ceil(CAST_THREADS as usize))?, 1, 1);
    // SAFETY: the kernel is the existing fixed BF16->F32 widening ABI, the
    // ranges cover exactly `elements`, and all owners outlive the enqueue.
    unsafe {
        launch_kernel(
            function.cu_function(),
            grid,
            (CAST_THREADS, 1, 1),
            0,
            stream.cu_stream(),
            &mut [
                arg(&mut source),
                arg(&mut destination),
                arg(&mut selected),
                arg(&mut count),
                arg(&mut width),
            ],
        )
    }
    .map_err(infer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_internal_metadata_failures_are_fatal_but_size_rejections_recover() {
        use crate::moe_api::Error;
        for error in [
            Error::Internal,
            Error::KernelFailure,
            Error::Malformed("plan"),
            Error::UnsupportedKernel(2),
            Error::UnsupportedAbi,
        ] {
            assert!(matches!(
                classify_metadata_error(error),
                ShapeError::Fatal(_)
            ));
        }
        for error in [
            Error::Overflow,
            Error::ShapeMismatch("rows"),
            Error::WorkspaceTooSmall,
            Error::InvalidArgument("rows"),
        ] {
            assert!(matches!(
                classify_metadata_error(error),
                ShapeError::Rejected(_)
            ));
        }
    }

    #[test]
    fn engine_geometry_guard_accepts_dynamic_abi_shapes_only() {
        assert!(geometry_supported(2048, 768));
        assert!(geometry_supported(4096, 1536));
        assert!(!geometry_supported(0, 768));
        assert!(!geometry_supported(2048, 0));
        assert!(!geometry_supported(2049, 768));
        assert!(!geometry_supported(2048, 769));
    }

    #[test]
    fn arch_mapping_matches_the_three_supported_policies() {
        assert_eq!(arch_for_compute_capability(9, 0).ok(), Some(90));
        assert_eq!(arch_for_compute_capability(10, 0).ok(), Some(100));
        assert_eq!(arch_for_compute_capability(12, 0).ok(), Some(120));
        assert!(arch_for_compute_capability(8, 0).is_err());
        assert!(arch_for_compute_capability(10, 1).is_err());
    }

    #[test]
    fn scratch_layout_counts_logits_and_aligns_workspace() {
        let layout = scratch_layout(7, 128, 4096).unwrap();
        assert_eq!(layout.tokens, 7);
        assert_eq!(layout.logits, 896);
        assert_eq!(layout.workspace, 4096);
        assert_eq!(layout.bytes, 896 * (BF16_BYTES + F32_BYTES) + 4096);
        assert_eq!(scratch_layout(1, 128, 1).unwrap().workspace, 256);
        assert_eq!(scratch_layout(1, 128, 256).unwrap().workspace, 256);
        assert_eq!(scratch_layout(1, 128, 257).unwrap().workspace, 512);
        assert!(scratch_layout(0, 128, 1).is_err());
        assert!(scratch_layout(usize::MAX, 2, 1).is_err());
        assert!(scratch_layout(1, 1, usize::MAX).is_err());
    }

    #[test]
    fn router_descriptor_is_token_exact_not_capacity_shaped() {
        let plan = LinearPlan::<Bf16>::contiguous(5, 2048, 128).unwrap();
        assert_eq!(plan.input_layout().shape(), &[5, 2048]);
        assert_eq!(plan.weight_layout().shape(), &[128, 2048]);
        assert_eq!(plan.output_layout().shape(), &[5, 128]);
    }

    #[test]
    fn actual_prefix_is_bounded_by_capacity_and_logits_slack() {
        assert_eq!(prefix_elements(4, 8, 16, 3, 24).ok(), Some((64, 12)));
        assert!(prefix_elements(9, 8, 16, 3, 27).is_err());
        assert!(prefix_elements(0, 8, 16, 3, 24).is_err());
        assert!(prefix_elements(4, 8, 16, 3, 11).is_err());
    }

    #[test]
    fn layer_element_products_are_canonical_and_overflow_checked() {
        let elements = layer_elements(128, 2048, 768).unwrap();
        assert_eq!(elements.router, 128 * 2048);
        assert_eq!(elements.gate_up, 128 * 1536 * 2048);
        assert_eq!(elements.down, 128 * 2048 * 768);
        assert!(layer_elements(usize::MAX, 2, 3).is_err());
        assert!(layer_elements(2, usize::MAX, 3).is_err());
    }
}
