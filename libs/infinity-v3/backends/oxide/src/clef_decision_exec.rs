// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Actual prepared GPU dataflow for the Native Clef joint head.
//!
//! The body mirrors `decision_models.cpp::ClefHead::score` in Native order:
//! hidden norm, span means/lexical gather, six projections, routing blocks,
//! attention-weighted field summaries, field decoder (self/cross/FFN), then
//! lexical prior, normalized cosine, residual scorer and gated joint logits.
//! It only launches borrowed cached Work buffers, the existing cuBLAS handle,
//! the selected FA4 provider and the small source-only Clef device leaf.

use crate::{
    clef_decision::infer, clef_decision_head::ClefHead, clef_decision_plan::ClefRequestPlan,
    clef_decision_prepared::ClefPrepared, clef_decision_workspace::ClefWorkspace, cublas::Blas,
    infer as backend_infer, tensor::Bf16,
};
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

fn checked_product(left: usize, right: usize, what: &'static str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| infer(format!("Clef {what} extent overflow")))
}

/// Leaf convenience entry: upload the prepared u32 metadata tail into the
/// cached Work metadata slot and enqueue the bound body.
///
/// # Safety
/// Same contract as [`enqueue_bound`]. The prepared host tail, cached Work and
/// output lease stay owned until the outer completion fence.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn enqueue(
    head: &ClefHead,
    hidden: &DeviceBuffer<Bf16>,
    lexical: &DeviceBuffer<Bf16>,
    workspace: &mut ClefWorkspace,
    prepared: &ClefPrepared,
    blas: &Blas,
    stream: &Stream,
    host_out: *mut f32,
) -> Result<(), BackendError> {
    let plan = prepared.plan();
    let bytes = plan
        .metadata
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| infer("Clef metadata byte size overflow"))?;
    cuda_core::memory::memcpy_htod_async(
        workspace.metadata.cu_deviceptr(),
        plan.metadata.as_ptr(),
        bytes,
        stream.cu_stream(),
    )
    .map_err(backend_infer)?;
    let metadata_base = workspace.metadata.cu_deviceptr();
    unsafe {
        enqueue_bound(
            head,
            hidden,
            lexical,
            workspace,
            prepared,
            blas,
            stream,
            host_out,
            metadata_base,
        )
    }
}

/// Enqueue every Clef kernel for one validated plan. No allocation, no stream
/// synchronization and no checkpoint read occurs here. `metadata_base` names
/// the caller's already-uploaded single Work input tail; the Q35 head worker
/// uses this entry so the leaf does not add a second H2D path.
///
/// # Safety
/// The engine bound this head's context on the current thread. `hidden`,
/// `lexical`, `workspace`, `blas` and the pinned `host_out` remain owned
/// without conflicting access until the caller's single completion fence
/// drains or quarantines the whole Work.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn enqueue_bound(
    head: &ClefHead,
    hidden: &DeviceBuffer<Bf16>,
    lexical: &DeviceBuffer<Bf16>,
    workspace: &mut ClefWorkspace,
    prepared: &ClefPrepared,
    blas: &Blas,
    stream: &Stream,
    host_out: *mut f32,
    metadata_base: u64,
) -> Result<(), BackendError> {
    // SAFETY: the wrapper preserves the outer-owner contract.
    unsafe {
        enqueue_bound_plan(
            head,
            hidden,
            lexical,
            workspace,
            prepared.plan(),
            blas,
            stream,
            host_out,
            metadata_base,
        )
    }
}

/// Launch using a CPU-only plan and the outer owner's uploaded input tail.
///
/// # Safety
/// The caller meets [`enqueue_bound`]'s allocation and completion contract.
/// `metadata_base` covers the retained device copy of the plan metadata. No
/// asynchronous transfer reads this CPU plan's Vec.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn enqueue_bound_plan(
    head: &ClefHead,
    hidden: &DeviceBuffer<Bf16>,
    lexical: &DeviceBuffer<Bf16>,
    workspace: &mut ClefWorkspace,
    plan: &ClefRequestPlan,
    blas: &Blas,
    stream: &Stream,
    host_out: *mut f32,
    metadata_base: u64,
) -> Result<(), BackendError> {
    let config = &head.config;
    let (h, width, ff) = (config.hidden, config.width, config.feedforward);
    let (tokens, requests, fields, options, option_tokens) = (
        plan.tokens,
        plan.requests,
        plan.fields,
        plan.options,
        plan.option_tokens,
    );
    let capacity = workspace.capacity;
    if tokens > capacity.tokens()
        || requests > capacity.requests()
        || fields > capacity.fields()
        || options > capacity.options()
        || option_tokens > capacity.option_tokens()
        || hidden.len() < checked_product(tokens, h, "hidden")?
        || lexical.len() < h
        || !lexical.len().is_multiple_of(h)
    {
        return Err(infer("Clef plan exceeds the retained Work capacity"));
    }
    if plan.metadata.len() > capacity.metadata_words() {
        return Err(infer(
            "Clef metadata tail exceeds the retained Work capacity",
        ));
    }
    let context = &head.context;
    if !Arc::ptr_eq(hidden.context(), context)
        || !Arc::ptr_eq(lexical.context(), context)
        || !Arc::ptr_eq(blas.context(), context)
        || !Arc::ptr_eq(workspace.metadata.context(), context)
        || !std::ptr::eq(blas.stream(), stream)
        || !std::ptr::eq(head.stream.as_ref(), stream)
    {
        return Err(infer(
            "Clef buffers belong to another CUDA context or stream",
        ));
    }
    let lexical_rows = lexical.len() / h;
    let ids = &plan.metadata[plan.layout.option_token_ids..plan.layout.total];
    if let Some(&id) = ids.iter().find(|&&id| id as usize >= lexical_rows) {
        return Err(infer(format!(
            "Clef option token id {id} is outside lexical rows 0..{lexical_rows}"
        )));
    }
    let mut run = crate::clef_decision_exec_state::ClefRun {
        head,
        metadata_base,
        workspace,
        plan,
        blas,
        stream,
        h,
        width,
        ff,
        tokens,
        requests,
        fields,
        options,
        option_tokens,
    };
    unsafe {
        crate::clef_decision_exec_inputs::inputs(&mut run, hidden, lexical)?;
        crate::clef_decision_exec_routing::routing(&mut run)?;
        crate::clef_decision_exec_decoder::decoder(&mut run)?;
        crate::clef_decision_exec_tail::tail(&mut run, host_out)?;
    }
    Ok(())
}
