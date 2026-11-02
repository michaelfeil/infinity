// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Small allocation-free launch helpers over the existing oxide kernels,
//! cuBLAS handle and FA4 typed borrowed views.

#![allow(clippy::too_many_arguments)]

use crate::{
    arg, blocks,
    clef_decision_head::{ClefKernels, ClefNorm},
    clef_decision_workspace::ClefWorkspace,
    cublas::Blas,
    infer, launch,
    math::LinearPlan,
    product,
    tensor::Bf16,
    u32_size,
};
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::BackendError;

pub(crate) fn element_pointer<T>(
    buffer: &DeviceBuffer<T>,
    elements: usize,
    what: &'static str,
) -> Result<u64, BackendError> {
    let offset = elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| infer(format!("Clef {what} byte offset overflow")))?;
    let offset =
        u64::try_from(offset).map_err(|_| infer(format!("Clef {what} byte offset exceeds u64")))?;
    buffer
        .cu_deviceptr()
        .checked_add(offset)
        .ok_or_else(|| infer(format!("Clef {what} address overflow")))
}

pub(crate) fn metadata_pointer(
    workspace: &ClefWorkspace,
    words: usize,
) -> Result<u64, BackendError> {
    element_pointer(&workspace.metadata, words, "metadata")
}

pub(crate) fn layer_norm(
    kernels: &ClefKernels,
    stream: &Stream,
    input: u64,
    norm: &ClefNorm,
    out: u64,
    rows: usize,
    width: usize,
) -> Result<(), BackendError> {
    let mut input = input;
    let mut residual = 0u64;
    let mut bias = 0u64;
    let mut gamma = norm.gamma.cu_deviceptr();
    let mut beta = norm.beta.cu_deviceptr();
    let mut out = out;
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    let mut eps = 1e-5f32;
    launch(
        &kernels.layer_norm,
        stream,
        (rows, 1, 1),
        0,
        &mut [
            arg(&mut input),
            arg(&mut residual),
            arg(&mut bias),
            arg(&mut gamma),
            arg(&mut beta),
            arg(&mut out),
            arg(&mut rows),
            arg(&mut width),
            arg(&mut eps),
        ],
    )
}

pub(crate) fn span_mean(
    kernels: &ClefKernels,
    stream: &Stream,
    input: u64,
    boundaries: u64,
    selected: u64,
    out: u64,
    count: usize,
    width: usize,
) -> Result<(), BackendError> {
    let mut input = input;
    let mut boundaries = boundaries;
    let mut selected = selected;
    let mut out = out;
    let mut count = u32_size(count)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.span_mean,
        stream,
        (count, 1, 1),
        0,
        &mut [
            arg(&mut input),
            arg(&mut boundaries),
            arg(&mut selected),
            arg(&mut out),
            arg(&mut count),
            arg(&mut width),
        ],
    )
}

pub(crate) fn embedding_gather(
    kernels: &ClefKernels,
    stream: &Stream,
    ids: u64,
    table: u64,
    out: u64,
    rows: usize,
    width: usize,
) -> Result<(), BackendError> {
    let mut ids = ids;
    let mut table = table;
    let mut out = out;
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.embedding_gather,
        stream,
        blocks(product(rows as usize, width as usize)?)?,
        0,
        &mut [
            arg(&mut ids),
            arg(&mut table),
            arg(&mut out),
            arg(&mut rows),
            arg(&mut width),
        ],
    )
}

pub(crate) unsafe fn linear(
    blas: &Blas,
    rows: usize,
    input: usize,
    output: usize,
    x: u64,
    weight: u64,
    y: u64,
) -> Result<(), BackendError> {
    let plan = LinearPlan::<Bf16>::contiguous(rows, input, output).map_err(infer)?;
    let arguments = plan.arguments(x, weight, y);
    // SAFETY: the caller owns every buffer through the outer completion fence.
    unsafe { blas.linear_bound(&arguments) }
}

/// Native `at::linear(x, weight, bias)` path: seed every output row with the
/// BF16 bias, then run one FP32-accumulating beta=1 GEMM. This matches the
/// `addmm`-style fused bias stage used by `DenseMlp` in Decision heads and
/// rounds to BF16 once, instead of a second BF16 post-GEMM bias add.
pub(crate) unsafe fn linear_biased(
    blas: &Blas,
    kernels: &ClefKernels,
    stream: &Stream,
    rows: usize,
    input: usize,
    output: usize,
    x: u64,
    weight: u64,
    bias: u64,
    y: u64,
) -> Result<(), BackendError> {
    let mut bias = bias;
    let mut y = y;
    let mut rows_u32 = u32_size(rows)?;
    let mut width_u32 = u32_size(output)?;
    launch(
        &kernels.bias_seed,
        stream,
        blocks(product(rows, output)?)?,
        0,
        &mut [
            arg(&mut bias),
            arg(&mut y),
            arg(&mut rows_u32),
            arg(&mut width_u32),
        ],
    )?;
    let plan = LinearPlan::<Bf16>::contiguous(rows, input, output).map_err(infer)?;
    let arguments = plan.arguments(x, weight, y);
    // SAFETY: the seed above is ordered before this beta=1 GEMM; the caller
    // retains every buffer and the seed kernel module through completion.
    unsafe { blas.enqueue_native_n1(&arguments) }
}

pub(crate) fn residual_add(
    kernels: &ClefKernels,
    stream: &Stream,
    a: u64,
    b: u64,
    out: u64,
    elements: usize,
) -> Result<(), BackendError> {
    let mut a = a;
    let mut b = b;
    let mut out = out;
    let mut elements = u32_size(elements)?;
    launch(
        &kernels.residual_add,
        stream,
        blocks(elements as usize)?,
        0,
        &mut [arg(&mut a), arg(&mut b), arg(&mut out), arg(&mut elements)],
    )
}

pub(crate) fn gelu_exact(
    kernels: &ClefKernels,
    stream: &Stream,
    input: &mut DeviceBuffer<Bf16>,
    elements: usize,
) -> Result<(), BackendError> {
    let mut input = input.cu_deviceptr();
    let mut elements = u32_size(elements)?;
    launch(
        &kernels.gelu_exact,
        stream,
        blocks(elements as usize)?,
        0,
        &mut [arg(&mut input), arg(&mut elements)],
    )
}
