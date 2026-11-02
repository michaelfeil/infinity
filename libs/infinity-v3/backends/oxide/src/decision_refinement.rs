// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Shared packed decision refinement blocks.
//!
//! Laya and (later) Clef use the same biasless-prenorm geometry as a native
//! Transformer encoder layer: affine LayerNorm, packed Q/K/V self attention,
//! residual add, and a biased feed-forward residual. Every biased projection is
//! the existing shared `BiasedLinear` policy, which accumulates in FP32 and
//! rounds to BF16 once (Lt epilogue or beta=1 over a bias-preseeded output).
//! There is no second BF16 bias add before attention, residual, GELU or logits.
//!
//! Oxide's ModernBERT route is BF16 storage with FP32 reductions. Laya head
//! tensors cross the same `WeightSource::read_bf16` boundary as the backbone.
//!
//! Laya's official H1024 head uses the shared A3 plain ATen-Welford
//! `infinity_plain_layer_norm_bf16_aten` entry when the cold head loaded it.
//! Its equality with Native ATen remains an explicit Root math gate.

use crate::{
    arg,
    biased_linear::{BiasedLinear, PreparedBiasedLinear},
    blocks,
    cublas::Blas,
    fa4, infer, launch, launch_threads,
    lt::{LtProvider, ShapeError},
    product,
    tensor::Bf16,
    u32_size,
};
use cuda_core::{DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;

pub(crate) struct AffineNorm {
    pub(crate) gamma: DeviceBuffer<Bf16>,
    pub(crate) beta: DeviceBuffer<Bf16>,
}

pub(crate) struct BiasedProjection {
    pub(crate) weight: DeviceBuffer<Bf16>,
    pub(crate) bias: DeviceBuffer<Bf16>,
    pub(crate) linear: BiasedLinear,
}

impl BiasedProjection {
    pub(crate) fn new(
        provider: &LtProvider,
        input: usize,
        output: usize,
        weight: DeviceBuffer<Bf16>,
        bias: DeviceBuffer<Bf16>,
    ) -> Result<Self, BackendError> {
        Ok(Self {
            weight,
            bias,
            linear: BiasedLinear::new(provider, input, output).map_err(ShapeError::into_backend)?,
        })
    }
}

pub(crate) struct RefinementBlock {
    pub(crate) norm1: AffineNorm,
    pub(crate) norm2: AffineNorm,
    pub(crate) qkv: BiasedProjection,
    pub(crate) out: BiasedProjection,
    pub(crate) ffn1: BiasedProjection,
    pub(crate) ffn2: BiasedProjection,
}

/// Validated packed row geometry. Projection descriptors live in the cold
/// `BiasedLinear` plans and are prepared per request before the lease begins.
pub(crate) struct RefinementPlan {
    pub(crate) tokens: usize,
    pub(crate) hidden: usize,
    pub(crate) ffn: usize,
}

impl RefinementPlan {
    pub(crate) fn new(tokens: usize, hidden: usize, ffn: usize) -> Result<Self, BackendError> {
        if tokens == 0 || hidden == 0 || ffn == 0 {
            return Err(infer(
                "decision refinement requires nonzero row and width geometry",
            ));
        }
        let _ = product(tokens, hidden)?;
        let _ = product(tokens, ffn)?;
        Ok(Self {
            tokens,
            hidden,
            ffn,
        })
    }
}

/// Caller-owned scratch for one refinement block. `state` is the residual
/// stream entering the block; the block leaves its result in `state`.
pub(crate) struct RefinementBuffers<'a> {
    pub(crate) state: &'a mut DeviceBuffer<Bf16>,
    pub(crate) next: &'a mut DeviceBuffer<Bf16>,
    pub(crate) norm: &'a mut DeviceBuffer<Bf16>,
    pub(crate) qkv: &'a mut DeviceBuffer<Bf16>,
    pub(crate) attention: &'a mut DeviceBuffer<Bf16>,
    pub(crate) branch: &'a mut DeviceBuffer<Bf16>,
    pub(crate) ffn: &'a mut DeviceBuffer<Bf16>,
}

pub(crate) struct RefinementKernels<'a> {
    pub(crate) layer_norm: &'a Function,
    pub(crate) layer_norm_aten: Option<&'a Function>,
    pub(crate) add: &'a Function,
    pub(crate) attention: &'a fa4::Fa4,
    pub(crate) relu: &'a Function,
}

/// Cold-biased projection selections for one block. Ownership stays with the
/// request until the single completion fence.
pub(crate) struct RefinementPrepared {
    pub(crate) qkv: PreparedBiasedLinear,
    pub(crate) out: PreparedBiasedLinear,
    pub(crate) ffn1: PreparedBiasedLinear,
    pub(crate) ffn2: PreparedBiasedLinear,
}

fn require_capacity(
    name: &'static str,
    buffer: &DeviceBuffer<Bf16>,
    elements: usize,
) -> Result<(), BackendError> {
    if buffer.len() < elements {
        return Err(infer(format!(
            "decision refinement {name} buffer has {} elements but needs {elements}",
            buffer.len()
        )));
    }
    Ok(())
}

pub(crate) fn layer_norm(
    kernel: &Function,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    norm: &AffineNorm,
    out: &DeviceBuffer<Bf16>,
    rows: u32,
    width: u32,
) -> Result<(), BackendError> {
    let mut input = input.cu_deviceptr();
    let mut residual = 0u64;
    let mut bias = 0u64;
    let mut gamma = norm.gamma.cu_deviceptr();
    let mut beta = norm.beta.cu_deviceptr();
    let mut out = out.cu_deviceptr();
    let mut rows = rows;
    let mut width = width;
    let mut eps = 1e-5f32;
    launch(
        kernel,
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

/// Plain affine ATen-Welford LayerNorm used by Laya H1024. The wrapper is the
/// shared A3 `infinity_plain_layer_norm_bf16_aten(input, gamma, beta, out,
/// rows, width, eps)` entry: nullable-free beta, explicit width, 128 threads
/// per row.
pub(crate) fn layer_norm_aten(
    kernel: &Function,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    norm: &AffineNorm,
    out: &DeviceBuffer<Bf16>,
    rows: u32,
    width: u32,
) -> Result<(), BackendError> {
    let mut input = input.cu_deviceptr();
    let mut gamma = norm.gamma.cu_deviceptr();
    let mut beta = norm.beta.cu_deviceptr();
    let mut out = out.cu_deviceptr();
    let mut rows = rows;
    let mut width = width;
    let mut eps = 1e-5f32;
    launch_threads(
        kernel,
        stream,
        (rows, 1, 1),
        128,
        0,
        &mut [
            arg(&mut input),
            arg(&mut gamma),
            arg(&mut beta),
            arg(&mut out),
            arg(&mut rows),
            arg(&mut width),
            arg(&mut eps),
        ],
    )
}

/// Pointwise activation over an already-biased BF16 projection.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn activation(
    kernel: &Function,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    out: &DeviceBuffer<Bf16>,
    rows: u32,
    width: u32,
) -> Result<(), BackendError> {
    let elements = usize::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_mul(width as usize))
        .ok_or_else(|| infer("decision refinement activation extent overflow"))?;
    let mut input = input.cu_deviceptr();
    let mut bias = 0u64;
    let mut out = out.cu_deviceptr();
    let mut rows = rows;
    let mut width = width;
    let mut stride = width;
    launch(
        kernel,
        stream,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut bias),
            arg(&mut out),
            arg(&mut rows),
            arg(&mut width),
            arg(&mut stride),
            arg(&mut stride),
        ],
    )
}

pub(crate) unsafe fn prepare_projection(
    projection: &mut BiasedProjection,
    provider: &LtProvider,
    blas: &Blas,
    rows: usize,
    stream: &Stream,
    x: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
) -> Result<PreparedBiasedLinear, BackendError> {
    unsafe {
        projection
            .linear
            .prepare(
                provider,
                blas,
                rows,
                stream,
                x,
                &projection.weight,
                &projection.bias,
                output,
            )
            .map_err(ShapeError::into_backend)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_projection(
    projection: &mut BiasedProjection,
    prepared: &PreparedBiasedLinear,
    provider: &LtProvider,
    blas: &Blas,
    seed: &Function,
    stream: &Stream,
    x: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<Bf16>,
) -> Result<(), BackendError> {
    unsafe {
        projection.linear.launch(
            prepared,
            provider,
            blas,
            seed,
            stream,
            x,
            &projection.weight,
            &projection.bias,
            output,
        )
    }
}

/// Bind every biased projection for this request before the lease begins.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn prepare_block(
    provider: &LtProvider,
    blas: &Blas,
    stream: &Stream,
    block: &mut RefinementBlock,
    plan: &RefinementPlan,
    buffers: &mut RefinementBuffers<'_>,
) -> Result<RefinementPrepared, BackendError> {
    let rows = plan.tokens;
    Ok(RefinementPrepared {
        qkv: unsafe {
            prepare_projection(
                &mut block.qkv,
                provider,
                blas,
                rows,
                stream,
                buffers.norm,
                buffers.qkv,
            )?
        },
        out: unsafe {
            prepare_projection(
                &mut block.out,
                provider,
                blas,
                rows,
                stream,
                buffers.attention,
                buffers.branch,
            )?
        },
        ffn1: unsafe {
            prepare_projection(
                &mut block.ffn1,
                provider,
                blas,
                rows,
                stream,
                buffers.norm,
                buffers.ffn,
            )?
        },
        ffn2: unsafe {
            prepare_projection(
                &mut block.ffn2,
                provider,
                blas,
                rows,
                stream,
                buffers.ffn,
                buffers.branch,
            )?
        },
    })
}

/// Enqueue one pre-norm self-attention + feed-forward refinement block over
/// packed rows isolated by `cumulative` boundaries.
///
/// # Safety
/// Every buffer belongs to the caller's bound context and stays owned without
/// conflicting access until the ordered stream drains. Prepared projections
/// were bound for these exact buffers before the request lease began.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn enqueue_block(
    provider: &LtProvider,
    blas: &Blas,
    seed: &Function,
    stream: &Stream,
    kernels: &RefinementKernels<'_>,
    block: &mut RefinementBlock,
    plan: &RefinementPlan,
    prepared: &RefinementPrepared,
    buffers: &mut RefinementBuffers<'_>,
    cumulative: u64,
    batch_boundaries: usize,
    heads: usize,
    max_length: usize,
) -> Result<(), BackendError> {
    let tokens = plan.tokens;
    let hidden = plan.hidden;
    let ffn = plan.ffn;
    if tokens == 0 || hidden == 0 || heads == 0 || !hidden.is_multiple_of(heads) || max_length == 0
    {
        return Err(infer("invalid decision refinement geometry"));
    }
    let head_dim = hidden / heads;
    let width = u32_size(hidden)?;
    let rows = u32_size(tokens)?;
    let batch = batch_boundaries
        .checked_sub(1)
        .ok_or_else(|| infer("decision refinement boundaries are incomplete"))?;
    let batch = u32_size(batch)?;
    let head_dim = u32_size(head_dim)?;
    let max_length = u32_size(max_length)?;
    let hidden_elements = product(tokens, hidden)?;
    let ffn_elements = product(tokens, ffn)?;
    for (name, buffer, needed) in [
        ("state", &*buffers.state, hidden_elements),
        ("next", &*buffers.next, hidden_elements),
        ("norm", &*buffers.norm, hidden_elements),
        ("qkv", &*buffers.qkv, product(hidden_elements, 3)?),
        ("attention", &*buffers.attention, hidden_elements),
        ("branch", &*buffers.branch, hidden_elements),
        ("ffn", &*buffers.ffn, ffn_elements),
    ] {
        require_capacity(name, buffer, needed)?;
    }
    if let Some(kernel) = kernels.layer_norm_aten {
        layer_norm_aten(
            kernel,
            stream,
            buffers.state,
            &block.norm1,
            buffers.norm,
            rows,
            width,
        )?;
    } else {
        layer_norm(
            kernels.layer_norm,
            stream,
            buffers.state,
            &block.norm1,
            buffers.norm,
            rows,
            width,
        )?;
    }
    unsafe {
        launch_projection(
            &mut block.qkv,
            &prepared.qkv,
            provider,
            blas,
            seed,
            stream,
            buffers.norm,
            buffers.qkv,
        )?;
    }
    // QKV already includes the Native at::linear bias. Use its packed
    // [tokens, 3, heads, head_dim] layout directly for FA4.
    let q = buffers.qkv.cu_deviceptr();
    let half_bytes = u64::try_from(hidden * std::mem::size_of::<Bf16>()).map_err(infer)?;
    let k = q
        .checked_add(half_bytes)
        .ok_or_else(|| infer("decision refinement K address overflow"))?;
    let v = k
        .checked_add(half_bytes)
        .ok_or_else(|| infer("decision refinement V address overflow"))?;
    let qkv_stride = i64::try_from(product(hidden, 3)?).map_err(infer)?;
    let row_stride = i64::from(width);
    let head_dim = i64::from(head_dim);
    let attention = fa4::AttentionBatch {
        q,
        k,
        v,
        out: buffers.attention.cu_deviceptr(),
        q_offsets: cumulative,
        k_offsets: cumulative,
        q_tokens: tokens,
        k_tokens: tokens,
        batch: usize::try_from(batch).map_err(infer)?,
        max_q: usize::try_from(max_length).map_err(infer)?,
        strides: [
            qkv_stride, head_dim, 1, qkv_stride, head_dim, 1, qkv_stride, head_dim, 1, row_stride,
            head_dim, 1,
        ],
    };
    unsafe { kernels.attention.forward_batch_bound(stream, &attention)? };
    unsafe {
        launch_projection(
            &mut block.out,
            &prepared.out,
            provider,
            blas,
            seed,
            stream,
            buffers.attention,
            buffers.branch,
        )?;
        let mut a = buffers.state.cu_deviceptr();
        let mut b = buffers.branch.cu_deviceptr();
        let mut out = buffers.next.cu_deviceptr();
        let mut elements = u32_size(hidden_elements)?;
        launch(
            kernels.add,
            stream,
            blocks(hidden_elements)?,
            0,
            &mut [arg(&mut a), arg(&mut b), arg(&mut out), arg(&mut elements)],
        )?;
    }
    if let Some(kernel) = kernels.layer_norm_aten {
        layer_norm_aten(
            kernel,
            stream,
            buffers.next,
            &block.norm2,
            buffers.norm,
            rows,
            width,
        )?;
    } else {
        layer_norm(
            kernels.layer_norm,
            stream,
            buffers.next,
            &block.norm2,
            buffers.norm,
            rows,
            width,
        )?;
    }
    unsafe {
        launch_projection(
            &mut block.ffn1,
            &prepared.ffn1,
            provider,
            blas,
            seed,
            stream,
            buffers.norm,
            buffers.ffn,
        )?;
        activation(
            kernels.relu,
            stream,
            buffers.ffn,
            buffers.ffn,
            rows,
            u32_size(ffn)?,
        )?;
        launch_projection(
            &mut block.ffn2,
            &prepared.ffn2,
            provider,
            blas,
            seed,
            stream,
            buffers.ffn,
            buffers.branch,
        )?;
        let mut a = buffers.next.cu_deviceptr();
        let mut b = buffers.branch.cu_deviceptr();
        let mut out = buffers.state.cu_deviceptr();
        let mut elements = u32_size(hidden_elements)?;
        launch(
            kernels.add,
            stream,
            blocks(hidden_elements)?,
            0,
            &mut [arg(&mut a), arg(&mut b), arg(&mut out), arg(&mut elements)],
        )
    }
}
