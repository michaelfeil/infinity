// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Clef field decoder self-attention, cross-attention and feed-forward.

use crate::{
    clef_decision_attention::qkv,
    clef_decision_exec_state::{checked_product, metadata_at, ClefRun},
    clef_decision_ops::{gelu_exact, layer_norm, linear_biased, residual_add},
};
use infinity_v3_backend_core::BackendError;

pub(crate) unsafe fn decoder(run: &mut ClefRun<'_>) -> Result<(), BackendError> {
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
    let _ = (h, options, option_tokens);
    let l = plan.layout;
    let memory = workspace.memory.cu_deviceptr();
    let request_field_offsets = metadata_at(metadata_base, l.request_field_offsets)?;
    let sequence_offsets = metadata_at(metadata_base, l.sequence_offsets)?;
    for decoder in &weights.decoder {
        layer_norm(
            kernels,
            stream,
            workspace.field_state.cu_deviceptr(),
            &decoder.n1,
            workspace.norm.cu_deviceptr(),
            fields,
            width,
        )?;
        qkv(
            kernels,
            blas,
            &head.fa4,
            stream,
            &decoder.self_attn,
            workspace,
            fields,
            fields,
            requests,
            plan.max_fields,
            workspace.norm.cu_deviceptr(),
            workspace.norm.cu_deviceptr(),
            request_field_offsets,
            request_field_offsets,
            width,
            head_dim,
        )?;
        residual_add(
            kernels,
            stream,
            workspace.field_state.cu_deviceptr(),
            workspace.branch.cu_deviceptr(),
            workspace.field_next.cu_deviceptr(),
            checked_product(fields, width, "decoder self residual")?,
        )?;
        layer_norm(
            kernels,
            stream,
            workspace.field_next.cu_deviceptr(),
            &decoder.n2,
            workspace.norm.cu_deviceptr(),
            fields,
            width,
        )?;
        qkv(
            kernels,
            blas,
            &head.fa4,
            stream,
            &decoder.cross,
            workspace,
            fields,
            tokens,
            requests,
            plan.max_fields,
            workspace.norm.cu_deviceptr(),
            memory,
            request_field_offsets,
            sequence_offsets,
            width,
            head_dim,
        )?;
        residual_add(
            kernels,
            stream,
            workspace.field_next.cu_deviceptr(),
            workspace.branch.cu_deviceptr(),
            workspace.field_state.cu_deviceptr(),
            checked_product(fields, width, "decoder cross residual")?,
        )?;
        layer_norm(
            kernels,
            stream,
            workspace.field_state.cu_deviceptr(),
            &decoder.n3,
            workspace.norm.cu_deviceptr(),
            fields,
            width,
        )?;
        unsafe {
            linear_biased(
                blas,
                kernels,
                stream,
                fields,
                width,
                ff,
                workspace.norm.cu_deviceptr(),
                decoder.linear1.weight.cu_deviceptr(),
                decoder.linear1.bias.cu_deviceptr(),
                workspace.ffn.cu_deviceptr(),
            )?;
        }
        gelu_exact(
            kernels,
            stream,
            &mut workspace.ffn,
            checked_product(fields, ff, "decoder ffn")?,
        )?;
        unsafe {
            linear_biased(
                blas,
                kernels,
                stream,
                fields,
                ff,
                width,
                workspace.ffn.cu_deviceptr(),
                decoder.linear2.weight.cu_deviceptr(),
                decoder.linear2.bias.cu_deviceptr(),
                workspace.branch.cu_deviceptr(),
            )?;
        }
        residual_add(
            kernels,
            stream,
            workspace.field_state.cu_deviceptr(),
            workspace.branch.cu_deviceptr(),
            workspace.field_state.cu_deviceptr(),
            checked_product(fields, width, "decoder residual")?,
        )?;
    }

    Ok(())
}
