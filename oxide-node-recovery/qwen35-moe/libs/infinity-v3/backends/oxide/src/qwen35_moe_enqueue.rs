// Typed routed-FFN enqueue preparation for Qwen3.5-MoE.
//
// This module owns no stream and queries nothing at enqueue time. The provider
// artifact supplies the architecture-specific workspace size and kernel plan
// (SM90 Hopper vs portable SM100/120). Every cross-context, length, geometry,
// workspace and workspace-alignment guard runs before the single ABI forward.
// No safe API accepts raw pointers; the caller must keep every typed owner and
// the bound stream alive until completion, and a failure is final (no retry or
// fallback).

use crate::cublas::Blas;
use crate::moe_abi::{Buffers, ABI_VERSION};
use crate::qwen35_moe_device::Qwen35MoeDeviceFfn;
use crate::qwen35_moe_provider::PreparedQwen35MoeRouted;
use crate::tensor::Bf16;
use crate::{arg, infer, launch, product, u32_size};
use cuda_core::{DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

const WORKSPACE_ALIGNMENT: usize = 256;
const CAST_THREADS: u32 = 256;

/// Provider-derived workspace capacity. The actual size comes from
/// `oxide_moe_query_workspace` for the selected architecture, so the Hopper ABI
/// is never approximated by the portable layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RoutedWorkspacePlan {
    bytes: usize,
}

impl RoutedWorkspacePlan {
    pub(crate) fn from_prepared(prepared: &PreparedQwen35MoeRouted) -> Result<Self, BackendError> {
        let bytes = prepared.workspace_bytes();
        if bytes == 0 || !bytes.is_multiple_of(WORKSPACE_ALIGNMENT) {
            return Err(infer(
                "provider workspace must be nonzero and 256-byte aligned",
            ));
        }
        Ok(Self { bytes })
    }

    pub(crate) const fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Typed buffers owned by the outer backbone/request. The routed enqueue
/// borrows them; it does not allocate, sort, read back or synchronize.
pub(crate) struct RoutedFfnBuffers<'a> {
    pub(crate) input: &'a DeviceBuffer<Bf16>,
    pub(crate) output: &'a mut DeviceBuffer<Bf16>,
    pub(crate) logits_bf16: &'a mut DeviceBuffer<Bf16>,
    pub(crate) logits_f32: &'a DeviceBuffer<f32>,
    pub(crate) workspace: &'a mut DeviceBuffer<u8>,
    pub(crate) stream: &'a Arc<Stream>,
}

fn same_context(actual: &Arc<cuda_core::Context>, expected: &Arc<cuda_core::Context>) -> bool {
    Arc::ptr_eq(actual, expected)
}

/// Enqueue router linear + BF16->F32 widening + one Native ABI forward.
///
/// # Safety
/// The caller must bind the provider context on this thread. Every typed buffer
/// and `layer` must belong to that context and stay alive without conflicting
/// access until `stream` drains. `workspace` must be at least
/// `RoutedWorkspacePlan::bytes()` and 256-byte aligned. The caller owns
/// completion; on error the whole stream must be drained or quarantined.
pub(crate) unsafe fn enqueue_routed_ffn(
    prepared: &PreparedQwen35MoeRouted,
    layer: &Qwen35MoeDeviceFfn,
    blas: &Blas,
    cast_bf16_to_f32: &Function,
    buffers: RoutedFfnBuffers<'_>,
    workspace_plan: RoutedWorkspacePlan,
) -> Result<(), BackendError> {
    let tokens = prepared.tokens();
    let hidden = prepared.hidden();
    let intermediate = prepared.intermediate();
    if tokens == 0
        || layer.hidden != hidden
        || layer.intermediate != intermediate
        || layer.experts != 256
        || prepared.shape().experts != 256
        || prepared.shape().top_k != 8
        || prepared.shape().hidden as usize != hidden
        || prepared.shape().intermediate as usize != intermediate
    {
        return Err(infer("routed FFN geometry differs from the prepared plan"));
    }
    if !same_context(blas.context(), buffers.stream.context()) {
        return Err(infer("routed FFN BLAS and stream contexts differ"));
    }
    if blas.stream().cu_stream() != buffers.stream.cu_stream() {
        return Err(infer("routed FFN BLAS and consumer streams differ"));
    }
    for actual in [
        buffers.input.context(),
        buffers.output.context(),
        buffers.logits_bf16.context(),
        buffers.logits_f32.context(),
        buffers.workspace.context(),
        layer.router.context(),
        layer.gate_up.context(),
        layer.down.context(),
    ] {
        if !same_context(actual, buffers.stream.context()) {
            return Err(infer("routed FFN buffer belongs to a different context"));
        }
    }
    let hidden_elements = product(tokens, hidden)?;
    let logits_elements = product(tokens, 256)?;
    if buffers.input.len() < hidden_elements
        || buffers.output.len() < hidden_elements
        || buffers.logits_bf16.len() < logits_elements
        || buffers.logits_f32.len() < logits_elements
        || buffers.workspace.len() < workspace_plan.bytes()
    {
        return Err(infer(
            "routed FFN buffer is shorter than the prepared prefix",
        ));
    }
    ensure_forward_admitted(prepared)?;
    let (expected_router, expected_gate_up, expected_down) =
        routed_weight_lengths(hidden, intermediate)?;
    if layer.router.len() != expected_router
        || layer.gate_up.len() != expected_gate_up
        || layer.down.len() != expected_down
    {
        return Err(infer(
            "routed FFN device weight lengths differ from canonical geometry",
        ));
    }

    let router_arguments = prepared
        .router_plan()
        .bind(buffers.input, &layer.router, buffers.logits_bf16)
        .map_err(infer)?;
    // SAFETY: contexts/lengths checked above; owners are retained by the caller.
    unsafe { blas.linear_bound(&router_arguments)? };

    let mut source = buffers.logits_bf16.cu_deviceptr();
    let mut destination = buffers.logits_f32.cu_deviceptr();
    let mut selected: u64 = 0;
    let mut count = u32_size(logits_elements)?;
    let mut width = 1u32;
    let grid = (
        u32_size(logits_elements.div_ceil(CAST_THREADS as usize))?,
        1,
        1,
    );
    // The fixed BF16->F32 widening ABI covers exactly `logits_elements`; all
    // owners outlive the enqueue.
    launch(
        cast_bf16_to_f32,
        buffers.stream,
        grid,
        0,
        &mut [
            arg(&mut source),
            arg(&mut destination),
            arg(&mut selected),
            arg(&mut count),
            arg(&mut width),
        ],
    )?;

    let abi_buffers = Buffers {
        struct_size: std::mem::size_of::<Buffers>() as u32,
        abi_version: ABI_VERSION,
        logits: buffers.logits_f32.cu_deviceptr() as *const f32,
        input_bf16: buffers.input.cu_deviceptr() as *const std::ffi::c_void,
        gate_up_weight_bf16: layer.gate_up.cu_deviceptr() as *const std::ffi::c_void,
        down_weight_bf16: layer.down.cu_deviceptr() as *const std::ffi::c_void,
        expert_scales: std::ptr::null(),
        output_bf16: buffers.output.cu_deviceptr() as *mut std::ffi::c_void,
        workspace: buffers.workspace.cu_deviceptr() as *mut std::ffi::c_void,
        workspace_bytes: workspace_plan.bytes(),
        stream: buffers.stream.cu_stream().cast(),
        flags: 0,
        reserved: [0; 3],
    };
    // SAFETY: the caller guaranteed context binding, owner lifetime and
    // workspace capacity; the retained plan matches this frame exactly.
    unsafe { prepared.plan().forward_bound(&abi_buffers) }.map_err(infer)
}

/// Canonical ABI1 Qwen35 routed weight lengths for `[E,H]`, `[E,2I,H]` and
/// `[E,H,I]`. The `2I` factor is explicit; this is deliberately pure so the
/// length guard is testable without a CUDA context.
pub(crate) fn routed_weight_lengths(
    hidden: usize,
    intermediate: usize,
) -> Result<(usize, usize, usize), BackendError> {
    let experts = 256usize;
    let router = product(experts, hidden)?;
    let gate_up = product(product(experts, product(intermediate, 2)?)?, hidden)?;
    let down = product(product(experts, hidden)?, intermediate)?;
    Ok((router, gate_up, down))
}

/// Compile-only providers (SM100/SM120) may expose CPU metadata and prepare a
/// plan, but must never be enqueued as though Root had qualified them.
pub(crate) fn ensure_forward_admitted(
    prepared: &PreparedQwen35MoeRouted,
) -> Result<(), BackendError> {
    if prepared.contract().compile_only() {
        return Err(infer(
            "compile-only provider cannot enqueue a routed FFN before Root GPU qualification",
        ));
    }
    Ok(())
}

/// Pure guard used by CPU tests: verifies the workspace plan is 256-aligned and
/// that the ABI descriptor size matches the mirrored C header.
pub(crate) fn validate_ffn_contract(plan: RoutedWorkspacePlan) -> Result<(), BackendError> {
    if plan.bytes() == 0 || !plan.bytes().is_multiple_of(WORKSPACE_ALIGNMENT) {
        return Err(infer("workspace plan is not 256-byte aligned"));
    }
    if std::mem::size_of::<Buffers>() != 96 || ABI_VERSION != 1 {
        return Err(infer("ABI1 Buffers layout drift"));
    }
    Ok(())
}
