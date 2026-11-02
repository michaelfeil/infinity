// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Clef hidden normalization, span/gather inputs and option-query projections.

use crate::{
    clef_decision_exec_state::{metadata_at, ClefRun},
    clef_decision_ops::{embedding_gather, layer_norm, linear, span_mean},
    clef_decision_ops_heads::query_compose,
    tensor::Bf16,
};
use cuda_core::DeviceBuffer;
use infinity_v3_backend_core::BackendError;

pub(crate) unsafe fn inputs(
    run: &mut ClefRun<'_>,
    hidden: &DeviceBuffer<Bf16>,
    lexical: &DeviceBuffer<Bf16>,
) -> Result<(), BackendError> {
    let head = run.head;
    let metadata_base = run.metadata_base;
    let workspace = &mut *run.workspace;
    let plan = run.plan;
    let blas = run.blas;
    let stream = run.stream;
    let (h, width) = (run.h, run.width);
    let (tokens, requests, fields, options, option_tokens) = (
        run.tokens,
        run.requests,
        run.fields,
        run.options,
        run.option_tokens,
    );
    let l = plan.layout;
    let question_boundaries = metadata_at(metadata_base, l.question_boundaries)?;
    let question_selected = metadata_at(metadata_base, l.question_selected)?;
    let option_boundaries = metadata_at(metadata_base, l.option_boundaries)?;
    let option_selected = metadata_at(metadata_base, l.option_selected)?;
    let lexical_boundaries = metadata_at(metadata_base, l.lexical_boundaries)?;
    let lexical_selected = metadata_at(metadata_base, l.lexical_selected)?;
    let last_rows = metadata_at(metadata_base, l.last_rows)?;
    let field_of_option = metadata_at(metadata_base, l.field_of_option)?;
    let option_token_ids = metadata_at(metadata_base, l.option_token_ids)?;
    let weights = head.weights();
    let kernels = &head.kernels;
    let hidden_norm = workspace.hidden_norm.cu_deviceptr();
    let memory = workspace.memory.cu_deviceptr();
    layer_norm(
        kernels,
        stream,
        hidden.cu_deviceptr(),
        &weights.hidden_norm,
        hidden_norm,
        tokens,
        h,
    )?;
    unsafe {
        linear(
            blas,
            tokens,
            h,
            width,
            hidden_norm,
            weights.memory_projection.weight.cu_deviceptr(),
            memory,
        )?;
    }
    embedding_gather(
        kernels,
        stream,
        last_rows,
        hidden_norm,
        workspace.global_rows.cu_deviceptr(),
        requests,
        h,
    )?;
    span_mean(
        kernels,
        stream,
        hidden_norm,
        question_boundaries,
        question_selected,
        workspace.question_mean.cu_deviceptr(),
        fields,
        h,
    )?;
    span_mean(
        kernels,
        stream,
        hidden_norm,
        option_boundaries,
        option_selected,
        workspace.option_context.cu_deviceptr(),
        options,
        h,
    )?;
    embedding_gather(
        kernels,
        stream,
        option_token_ids,
        lexical.cu_deviceptr(),
        workspace.lexical_rows.cu_deviceptr(),
        option_tokens,
        h,
    )?;
    span_mean(
        kernels,
        stream,
        workspace.lexical_rows.cu_deviceptr(),
        lexical_boundaries,
        lexical_selected,
        workspace.lexical_mean.cu_deviceptr(),
        options,
        h,
    )?;
    unsafe {
        linear(
            blas,
            fields,
            h,
            width,
            workspace.question_mean.cu_deviceptr(),
            weights.question_projection.weight.cu_deviceptr(),
            workspace.question_base.cu_deviceptr(),
        )?;
        linear(
            blas,
            fields,
            h,
            width,
            workspace.question_mean.cu_deviceptr(),
            weights.option_question_projection.weight.cu_deviceptr(),
            workspace.question_query.cu_deviceptr(),
        )?;
        linear(
            blas,
            options,
            h,
            width,
            workspace.option_context.cu_deviceptr(),
            weights.option_context_projection.weight.cu_deviceptr(),
            workspace.context_projection.cu_deviceptr(),
        )?;
        linear(
            blas,
            options,
            h,
            width,
            workspace.lexical_mean.cu_deviceptr(),
            weights.option_lexical_projection.weight.cu_deviceptr(),
            workspace.lexical_projection.cu_deviceptr(),
        )?;
    }
    query_compose(
        kernels,
        stream,
        workspace.question_query.cu_deviceptr(),
        workspace.context_projection.cu_deviceptr(),
        workspace.lexical_projection.cu_deviceptr(),
        field_of_option,
        workspace.queries.cu_deviceptr(),
        options,
        width,
    )?;

    Ok(())
}
