// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Clef final field norm, prior/cosine features, scorer and logit readback.

use crate::{
    clef_decision::infer,
    clef_decision_exec_state::{checked_product, metadata_at, ClefRun},
    clef_decision_ops::{gelu_exact, layer_norm, linear_biased},
    clef_decision_ops_heads::{anchor, features, logits, prior},
};
use infinity_v3_backend_core::BackendError;

pub(crate) unsafe fn tail(run: &mut ClefRun<'_>, host_out: *mut f32) -> Result<(), BackendError> {
    let head = run.head;
    let metadata_base = run.metadata_base;
    let workspace = &mut *run.workspace;
    let plan = run.plan;
    let blas = run.blas;
    let stream = run.stream;
    let (h, width, ff) = (run.h, run.width, run.ff);
    let (tokens, requests, fields, options, option_tokens) = (
        run.tokens,
        run.requests,
        run.fields,
        run.options,
        run.option_tokens,
    );
    let weights = head.weights();
    let kernels = &head.kernels;
    let _ = (ff, tokens, requests, option_tokens);
    let l = plan.layout;
    let field_request = metadata_at(metadata_base, l.field_request)?;
    let field_of_option = metadata_at(metadata_base, l.field_of_option)?;
    layer_norm(
        kernels,
        stream,
        workspace.field_state.cu_deviceptr(),
        &weights.field_norm,
        workspace.field_norm.cu_deviceptr(),
        fields,
        width,
    )?;
    anchor(
        kernels,
        stream,
        workspace.question_mean.cu_deviceptr(),
        workspace.global_rows.cu_deviceptr(),
        field_request,
        workspace.anchors.cu_deviceptr(),
        fields,
        h,
    )?;
    prior(
        kernels,
        stream,
        workspace.lexical_mean.cu_deviceptr(),
        workspace.anchors.cu_deviceptr(),
        field_of_option,
        workspace.prior.cu_deviceptr(),
        options,
        h,
        weights.prior_scale,
    )?;
    layer_norm(
        kernels,
        stream,
        workspace.queries.cu_deviceptr(),
        &weights.option_norm,
        workspace.option_norm.cu_deviceptr(),
        options,
        width,
    )?;
    features(
        kernels,
        stream,
        workspace.field_norm.cu_deviceptr(),
        workspace.option_norm.cu_deviceptr(),
        field_of_option,
        workspace.residual_features.cu_deviceptr(),
        workspace.cosine.cu_deviceptr(),
        options,
        width,
    )?;
    unsafe {
        linear_biased(
            blas,
            kernels,
            stream,
            options,
            4 * width,
            width,
            workspace.residual_features.cu_deviceptr(),
            weights.scorer1.weight.cu_deviceptr(),
            weights.scorer1.bias.cu_deviceptr(),
            workspace.state.cu_deviceptr(),
        )?;
    }
    gelu_exact(
        kernels,
        stream,
        &mut workspace.state,
        checked_product(options, width, "scorer")?,
    )?;
    unsafe {
        linear_biased(
            blas,
            kernels,
            stream,
            options,
            width,
            1,
            workspace.state.cu_deviceptr(),
            weights.scorer3.weight.cu_deviceptr(),
            weights.scorer3.bias.cu_deviceptr(),
            workspace.residual.cu_deviceptr(),
        )?;
    }
    logits(
        kernels,
        stream,
        workspace.prior.cu_deviceptr(),
        workspace.cosine.cu_deviceptr(),
        workspace.residual.cu_deviceptr(),
        workspace.logits_f32.cu_deviceptr(),
        options,
        weights.joint_scale,
        weights.gate,
    )?;
    let bytes = options
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| infer("Clef logit readback size overflow"))?;
    cuda_core::memory::memcpy_dtoh_async(
        host_out,
        workspace.logits_f32.cu_deviceptr(),
        bytes,
        stream.cu_stream(),
    )
    .map_err(infer)?;
    Ok(())
}
