// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Clef routing refinement and attention-weighted field summaries.

use crate::{
    clef_decision_attention::qkv,
    clef_decision_exec_state::{checked_product, metadata_at, ClefRun},
    clef_decision_ops::{gelu_exact, layer_norm, linear, linear_biased, residual_add},
    clef_decision_ops_heads::{field_compose, route_scores, softmax_sum},
};
use infinity_v3_backend_core::BackendError;

pub(crate) unsafe fn routing(run: &mut ClefRun<'_>) -> Result<(), BackendError> {
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
    let head_dim = head.config.head_dim();
    let _ = option_tokens;
    let l = plan.layout;
    let memory = workspace.memory.cu_deviceptr();
    let request_option_offsets = metadata_at(metadata_base, l.request_option_offsets)?;
    let sequence_offsets = metadata_at(metadata_base, l.sequence_offsets)?;
    let field_of_option = metadata_at(metadata_base, l.field_of_option)?;
    let field_option_offsets = metadata_at(metadata_base, l.field_option_offsets)?;
    let field_request = metadata_at(metadata_base, l.field_request)?;
    let field_kind = metadata_at(metadata_base, l.field_kind)?;
    let scale = 1.0 / (width as f32).sqrt();
    for routing in &weights.routing {
        layer_norm(
            kernels,
            stream,
            workspace.queries.cu_deviceptr(),
            &routing.query,
            workspace.norm.cu_deviceptr(),
            options,
            width,
        )?;
        layer_norm(
            kernels,
            stream,
            memory,
            &routing.memory,
            workspace.memory_norm.cu_deviceptr(),
            tokens,
            width,
        )?;
        qkv(
            kernels,
            blas,
            &head.fa4,
            stream,
            &routing.attention,
            workspace,
            options,
            tokens,
            requests,
            plan.max_options,
            workspace.queries.cu_deviceptr(),
            workspace.memory_norm.cu_deviceptr(),
            request_option_offsets,
            sequence_offsets,
            width,
            head_dim,
        )?;
        residual_add(
            kernels,
            stream,
            workspace.queries.cu_deviceptr(),
            workspace.branch.cu_deviceptr(),
            workspace.next.cu_deviceptr(),
            checked_product(options, width, "routing attention residual")?,
        )?;
        layer_norm(
            kernels,
            stream,
            workspace.next.cu_deviceptr(),
            &routing.feed,
            workspace.norm.cu_deviceptr(),
            options,
            width,
        )?;
        unsafe {
            linear_biased(
                blas,
                kernels,
                stream,
                options,
                width,
                ff,
                workspace.norm.cu_deviceptr(),
                routing.ffn1.weight.cu_deviceptr(),
                routing.ffn1.bias.cu_deviceptr(),
                workspace.ffn.cu_deviceptr(),
            )?;
        }
        gelu_exact(
            kernels,
            stream,
            &mut workspace.ffn,
            checked_product(options, ff, "ffn")?,
        )?;
        unsafe {
            linear_biased(
                blas,
                kernels,
                stream,
                options,
                ff,
                width,
                workspace.ffn.cu_deviceptr(),
                routing.ffn2.weight.cu_deviceptr(),
                routing.ffn2.bias.cu_deviceptr(),
                workspace.branch.cu_deviceptr(),
            )?;
        }
        residual_add(
            kernels,
            stream,
            workspace.next.cu_deviceptr(),
            workspace.branch.cu_deviceptr(),
            workspace.queries.cu_deviceptr(),
            checked_product(options, width, "routing")?,
        )?;
    }
    route_scores(
        kernels,
        stream,
        workspace.queries.cu_deviceptr(),
        workspace.question_base.cu_deviceptr(),
        field_of_option,
        workspace.scores.cu_deviceptr(),
        options,
        width,
        fields,
        scale,
    )?;
    softmax_sum(
        kernels,
        stream,
        workspace.queries.cu_deviceptr(),
        workspace.scores.cu_deviceptr(),
        field_option_offsets,
        workspace.summaries.cu_deviceptr(),
        options,
        width,
        fields,
    )?;
    layer_norm(
        kernels,
        stream,
        workspace.summaries.cu_deviceptr(),
        &weights.option_summary_norm,
        workspace.summary_norm.cu_deviceptr(),
        fields,
        width,
    )?;
    unsafe {
        linear(
            blas,
            requests,
            h,
            width,
            workspace.global_rows.cu_deviceptr(),
            weights.global_projection.weight.cu_deviceptr(),
            workspace.global_projection.cu_deviceptr(),
        )?;
    }
    field_compose(
        kernels,
        stream,
        workspace.question_base.cu_deviceptr(),
        workspace.summary_norm.cu_deviceptr(),
        workspace.global_projection.cu_deviceptr(),
        weights.types.cu_deviceptr(),
        field_request,
        field_kind,
        workspace.field_state.cu_deviceptr(),
        fields,
        width,
    )?;

    Ok(())
}
