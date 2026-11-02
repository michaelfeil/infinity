// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Clef attention projection composition over FA4 typed borrowed views.

use crate::{
    clef_decision_head::ClefAttention,
    clef_decision_ops::{element_pointer, linear_biased},
    clef_decision_ops_attention::attention as enqueue_attention,
    clef_decision_workspace::ClefWorkspace,
    cublas::Blas,
    fa4::Fa4,
    infer,
};
use cuda_core::Stream;
use infinity_v3_backend_core::BackendError;

fn slice_weight(
    attention: &ClefAttention,
    index: usize,
    width: usize,
) -> Result<u64, BackendError> {
    let elements = index
        .checked_mul(
            width
                .checked_mul(width)
                .ok_or_else(|| infer("Clef qkv slice overflow"))?,
        )
        .ok_or_else(|| infer("Clef qkv slice overflow"))?;
    element_pointer(&attention.qkv, elements, "qkv slice")
}

/// Native `Attention` with separate Q/K/V projections and optional cross memory.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qkv(
    kernels: &crate::clef_decision_head::ClefKernels,
    blas: &Blas,
    fa4: &Fa4,
    stream: &Stream,
    attn: &ClefAttention,
    workspace: &mut ClefWorkspace,
    q_rows: usize,
    kv_rows: usize,
    requests: usize,
    max_q: usize,
    q_input: u64,
    kv_input: u64,
    q_offsets: u64,
    k_offsets: u64,
    width: usize,
    head_dim: usize,
) -> Result<(), BackendError> {
    unsafe {
        linear_biased(
            blas,
            kernels,
            stream,
            q_rows,
            width,
            width,
            q_input,
            slice_weight(attn, 0, width)?,
            element_pointer(&attn.bias, 0, "q bias")?,
            workspace.query.cu_deviceptr(),
        )?;
        linear_biased(
            blas,
            kernels,
            stream,
            kv_rows,
            width,
            width,
            kv_input,
            slice_weight(attn, 1, width)?,
            element_pointer(&attn.bias, width, "k bias")?,
            workspace.key.cu_deviceptr(),
        )?;
        linear_biased(
            blas,
            kernels,
            stream,
            kv_rows,
            width,
            width,
            kv_input,
            slice_weight(attn, 2, width)?,
            element_pointer(&attn.bias, 2 * width, "v bias")?,
            workspace.value.cu_deviceptr(),
        )?;
    }
    unsafe {
        enqueue_attention(
            fa4,
            stream,
            workspace.query.cu_deviceptr(),
            workspace.key.cu_deviceptr(),
            workspace.value.cu_deviceptr(),
            workspace.attention.cu_deviceptr(),
            q_offsets,
            k_offsets,
            q_rows,
            kv_rows,
            requests,
            max_q,
            width,
            width / head_dim,
            head_dim,
        )?;
    }
    unsafe {
        linear_biased(
            blas,
            kernels,
            stream,
            q_rows,
            width,
            width,
            workspace.attention.cu_deviceptr(),
            attn.out.weight.cu_deviceptr(),
            attn.out.bias.cu_deviceptr(),
            workspace.branch.cu_deviceptr(),
        )?;
    }
    Ok(())
}
