// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! FA4 packed attention binding for Clef query/memory rows.

#![allow(clippy::too_many_arguments)]

use crate::{
    fa4::{AttentionBatch, Fa4},
    infer,
};
use cuda_core::Stream;
use infinity_v3_backend_core::BackendError;

pub(crate) unsafe fn attention(
    fa4: &Fa4,
    stream: &Stream,
    q: u64,
    k: u64,
    v: u64,
    out: u64,
    q_offsets: u64,
    k_offsets: u64,
    q_tokens: usize,
    k_tokens: usize,
    batch: usize,
    max_q: usize,
    width: usize,
    heads: usize,
    head_dim: usize,
) -> Result<(), BackendError> {
    let strides = [
        i64::try_from(width).map_err(infer)?,
        i64::try_from(head_dim).map_err(infer)?,
        1,
        i64::try_from(width).map_err(infer)?,
        i64::try_from(head_dim).map_err(infer)?,
        1,
        i64::try_from(width).map_err(infer)?,
        i64::try_from(head_dim).map_err(infer)?,
        1,
        i64::try_from(width).map_err(infer)?,
        i64::try_from(head_dim).map_err(infer)?,
        1,
    ];
    let _ = heads;
    let batch = AttentionBatch {
        q,
        k,
        v,
        out,
        q_offsets,
        k_offsets,
        q_tokens,
        k_tokens,
        batch,
        max_q,
        strides,
    };
    // SAFETY: the outer engine bound the context; all owners stay in the cached
    // Work and the metadata tail until the one completion fence.
    unsafe { fa4.forward_batch_bound(stream, &batch) }
}
