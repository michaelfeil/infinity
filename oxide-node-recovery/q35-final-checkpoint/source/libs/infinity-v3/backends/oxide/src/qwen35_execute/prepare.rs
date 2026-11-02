// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold binding keeps the native individual projection widths unchanged.
use super::*;
use crate::fa4::{AttentionSpec, DType, Scale, Window};
use crate::qwen35_block_attention::{
    BlockArchitecture, BlockBoundaryBuffers, BlockLayout, BlockSequence, BlockTensor,
    PreparedBlockPlan,
};
use std::sync::Arc;

impl PreparedWork {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        weights: &Qwen35DeviceWeights,
        layout: &Qwen35Layout,
        slabs: &Slabs<'_>,
        tokens: usize,
        positions: u64,
        cumulative: u64,
        sequences: usize,
        max_len: usize,
    ) -> Result<Self, BackendError> {
        let c = &weights.config;
        c.validate_candidate()?;
        let geometry = AttentionGeometry::from_config(c);
        let mut attention_layers = Vec::new();
        attention_layers
            .try_reserve_exact(weights.layers.len())
            .map_err(infer)?;
        for layer in &weights.layers {
            attention_layers.push(AttentionLayerWeights {
                attention: &layer.attention,
                input_norm: &layer.input_norm,
                post_attention_norm: &layer.post_attention_norm,
            });
        }
        let attention = PreparedAttentionWork::new(
            &geometry,
            &attention_layers,
            &weights.embed,
            &weights.final_norm,
            layout,
            slabs,
            tokens,
            positions,
            cumulative,
            sequences,
            max_len,
        )?;
        let h = c.hidden_size;
        let i = c.intermediate_size;
        let context = slabs.residual.context();
        let hp = slabs.residual.cu_deviceptr();
        let ap = slabs.projection.cu_deviceptr();
        let bp = slabs.activation.cu_deviceptr();
        let a = |offset, width| span(ap, slabs.projection.len(), offset, product(tokens, width)?);
        let b = |offset, width| span(bp, slabs.activation.len(), offset, product(tokens, width)?);
        let norm = attention.norm_address();
        let output = attention.output_address();
        let mut ffn = Vec::new();
        ffn.try_reserve_exact(weights.layers.len()).map_err(infer)?;
        let mut next = Vec::new();
        next.try_reserve_exact(weights.layers.len())
            .map_err(infer)?;
        for (index, layer) in weights.layers.iter().enumerate() {
            for buffer in [&layer.gate_up, &layer.down] {
                if !Arc::ptr_eq(context, buffer.context()) {
                    return Err(infer("Qwen3.5 layer context mismatch"));
                }
            }
            ffn.push(FfnStep {
                gate_up: linear(
                    tokens,
                    h,
                    product(i, 2)?,
                    norm,
                    layer.gate_up.cu_deviceptr(),
                    a(0, product(i, 2)?)?,
                )?,
                down: linear(tokens, i, h, b(0, i)?, layer.down.cu_deviceptr(), output)?,
            });
            next.push(ResidualOp::new(
                hp,
                output,
                attention.next_gamma(index)?,
                norm,
                tokens,
                h,
            )?);
        }
        Ok(Self {
            attention,
            ffn,
            next,
            intermediate: u32_size(i)?,
        })
    }

    /// Build the ordinary prepared frame plus one checked cold block plan.
    ///
    /// The block plan reuses the full-attention Q/K/V/output addresses bound by
    /// [`Self::new`]; it adds no device storage. `prefix` is measured in packed
    /// tokens and must address the single sequence's first `prefix` rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_block(
        weights: &Qwen35DeviceWeights,
        layout: &Qwen35Layout,
        slabs: &Slabs<'_>,
        tokens: usize,
        positions: u64,
        cumulative: u64,
        sequences: usize,
        max_len: usize,
        prefix: usize,
        architecture: BlockArchitecture,
        device_prefix: u64,
        device_suffix: u64,
        device_combined: u64,
    ) -> Result<Self, BackendError> {
        if sequences != 1 {
            return Err(infer(
                "Qwen3.5 prepared block attention admits exactly one packed sequence",
            ));
        }
        let mut prepared = Self::new(
            weights, layout, slabs, tokens, positions, cumulative, sequences, max_len,
        )?;
        let c = &weights.config;
        let spec = AttentionSpec {
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
            causal: false,
            window: Window::Global,
            dtype: DType::Bf16,
            scale: Scale::RsqrtHeadDim,
        };
        let capacity_tokens = layout.capacity_tokens();
        let batch = prepared.attention.batch();
        let block_layout = BlockLayout {
            q: BlockTensor {
                address: batch.q,
                capacity_tokens,
            },
            k: BlockTensor {
                address: batch.k,
                capacity_tokens,
            },
            v: BlockTensor {
                address: batch.v,
                capacity_tokens,
            },
            out: BlockTensor {
                address: batch.out,
                capacity_tokens,
            },
        };
        let block = PreparedBlockPlan::prepare(
            architecture,
            &spec,
            block_layout,
            &[BlockSequence::new(prefix, tokens)],
            BlockBoundaryBuffers {
                prefix: device_prefix,
                suffix: device_suffix,
                combined: device_combined,
                capacity_entries: 2,
            },
            0,
        )?;
        prepared.attention.set_block(block);
        Ok(prepared)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_attention_work(
    geometry: &AttentionGeometry,
    layers: &[AttentionLayerWeights<'_>],
    embed: &DeviceBuffer<Bf16>,
    final_norm: &DeviceBuffer<Bf16>,
    layout: &impl crate::qwen35_layout::AttentionSlabLayout,
    slabs: &Slabs<'_>,
    tokens: usize,
    positions: u64,
    cumulative: u64,
    sequences: usize,
    max_len: usize,
) -> Result<PreparedAttentionWork, BackendError> {
    let c = geometry;
    if tokens == 0
        || tokens > layout.capacity_tokens()
        || sequences == 0
        || sequences > tokens
        || max_len == 0
        || max_len > tokens
        || positions == 0
        || cumulative == 0
        || !positions.is_multiple_of(4)
        || !cumulative.is_multiple_of(4)
        || layers.len() != c.num_hidden_layers
    {
        return Err(infer("Qwen3.5 invalid actual prepared frame geometry"));
    }
    let context = slabs.residual.context();
    for buffer in [slabs.projection, slabs.activation, embed, final_norm] {
        if !Arc::ptr_eq(context, buffer.context()) {
            return Err(infer("Qwen3.5 phase/weight context mismatch"));
        }
    }
    let h = c.hidden_size;
    let q = c.q_width()?;
    let kv = c.kv_width()?;
    let vh = c.linear_num_value_heads;
    let v = product(vh, 128)?;
    let channels = product(
        c.linear_num_key_heads
            .checked_mul(2)
            .and_then(|n| n.checked_add(vh))
            .ok_or_else(|| infer("GDN channels overflow"))?,
        128,
    )?;
    let cap = layout.capacity_tokens();
    if slabs.residual.len() < layout.hidden_elements()
        || slabs.projection.len() < layout.projection_elements()
        || slabs.activation.len() < layout.activation_elements()
    {
        return Err(infer("Qwen3.5 slab capacity mismatch"));
    }
    let hp = slabs.residual.cu_deviceptr();
    let ap = slabs.projection.cu_deviceptr();
    let bp = slabs.activation.cu_deviceptr();
    let a = |offset, width| span(ap, slabs.projection.len(), offset, product(tokens, width)?);
    let b = |offset, width| span(bp, slabs.activation.len(), offset, product(tokens, width)?);
    let norm = b(0, h)?;
    let output = a(0, h)?;
    let qg = a(0, product(q, 2)?)?;
    let key = a(product(cap, product(q, 2)?)?, kv)?;
    let value = a(
        product(
            cap,
            product(q, 2)?
                .checked_add(kv)
                .ok_or_else(|| infer("QKV offset overflow"))?,
        )?,
        kv,
    )?;
    let query = b(0, q)?;
    let normalized_key = b(product(cap, q)?, kv)?;
    let attended = b(
        product(
            cap,
            q.checked_add(kv)
                .ok_or_else(|| infer("Attention offset overflow"))?,
        )?,
        q,
    )?;
    let qkv = a(0, channels)?;
    let zo = product(cap, h)?;
    let z = b(zo, v)?;
    let abo = product(
        cap,
        h.checked_add(v)
            .ok_or_else(|| infer("GDN AB offset overflow"))?,
    )?;
    let ab_width = product(vh, 2)?;
    let ab = b(abo, ab_width)?;
    let ato = abo
        .checked_add(product(cap, ab_width)?)
        .ok_or_else(|| infer("GDN A offset overflow"))?;
    let at = b(ato, vh)?;
    let bt = b(
        ato.checked_add(product(cap, vh)?)
            .ok_or_else(|| infer("GDN B offset overflow"))?,
        vh,
    )?;
    let first = layers
        .first()
        .ok_or_else(|| infer("Qwen3.5 has no layers"))?;
    let initial = NormOp::new(
        hp,
        first.input_norm.cu_deviceptr(),
        norm,
        tokens,
        h,
        1,
        h,
        0,
    )?;
    let mut prepared_layers = Vec::new();
    prepared_layers
        .try_reserve_exact(layers.len())
        .map_err(infer)?;
    let mut next_gammas = Vec::new();
    next_gammas.try_reserve_exact(layers.len()).map_err(infer)?;
    for (index, layer) in layers.iter().enumerate() {
        for buffer in [layer.input_norm, layer.post_attention_norm] {
            if !Arc::ptr_eq(context, buffer.context()) {
                return Err(infer("Qwen3.5 layer context mismatch"));
            }
        }
        let attention = match layer.attention {
            Qwen35DeviceAttention::Full {
                qkv: matrix,
                output: ow,
                q_norm,
                k_norm,
            } => {
                for buffer in [matrix, ow, q_norm, k_norm] {
                    if !Arc::ptr_eq(context, buffer.context()) {
                        return Err(infer("Qwen3.5 full weight context mismatch"));
                    }
                }
                let matrix = matrix.cu_deviceptr();
                AttentionStep::Full {
                    qg: linear(tokens, h, product(q, 2)?, norm, matrix, qg)?,
                    key: linear(
                        tokens,
                        h,
                        kv,
                        norm,
                        pointer(matrix, product(product(q, 2)?, h)?)?,
                        key,
                    )?,
                    value: linear(
                        tokens,
                        h,
                        kv,
                        norm,
                        pointer(
                            matrix,
                            product(
                                product(q, 2)?
                                    .checked_add(kv)
                                    .ok_or_else(|| infer("QKV weight offset overflow"))?,
                                h,
                            )?,
                        )?,
                        value,
                    )?,
                    output: linear(tokens, q, h, attended, ow.cu_deviceptr(), output)?,
                    q_norm: NormOp::new(
                        qg,
                        q_norm.cu_deviceptr(),
                        query,
                        product(tokens, c.num_attention_heads)?,
                        c.head_dim,
                        c.num_attention_heads,
                        product(q, 2)?,
                        product(c.head_dim, 2)?,
                    )?,
                    k_norm: NormOp::new(
                        key,
                        k_norm.cu_deviceptr(),
                        normalized_key,
                        product(tokens, c.num_key_value_heads)?,
                        c.head_dim,
                        c.num_key_value_heads,
                        kv,
                        c.head_dim,
                    )?,
                }
            }
            Qwen35DeviceAttention::Linear {
                qkv: matrix,
                z: zw,
                ab: abw,
                output: ow,
                ..
            } => {
                for buffer in [matrix, zw, abw, ow] {
                    if !Arc::ptr_eq(context, buffer.context()) {
                        return Err(infer("Qwen3.5 linear weight context mismatch"));
                    }
                }
                AttentionStep::Linear {
                    qkv: linear(tokens, h, channels, norm, matrix.cu_deviceptr(), qkv)?,
                    z: linear(tokens, h, v, norm, zw.cu_deviceptr(), z)?,
                    a: linear(tokens, h, vh, norm, abw.cu_deviceptr(), at)?,
                    b: linear(
                        tokens,
                        h,
                        vh,
                        norm,
                        pointer(abw.cu_deviceptr(), product(vh, h)?)?,
                        bt,
                    )?,
                    output: linear(tokens, v, h, ab, ow.cu_deviceptr(), output)?,
                }
            }
        };
        let next_gamma = layers
            .get(index + 1)
            .map_or(final_norm, |next| next.input_norm)
            .cu_deviceptr();
        next_gammas.push(next_gamma);
        prepared_layers.push(AttentionLayerStep {
            attention,
            post: ResidualOp::new(
                hp,
                output,
                layer.post_attention_norm.cu_deviceptr(),
                norm,
                tokens,
                h,
            )?,
        });
    }
    let stride = |width| i64::try_from(width).map_err(infer);
    Ok(PreparedAttentionWork {
        first: initial,
        layers: prepared_layers,
        frame: AttentionFrame {
            tokens: u32_size(tokens)?,
            qheads: u32_size(c.num_attention_heads)?,
            kvheads: u32_size(c.num_key_value_heads)?,
            value_heads: u32_size(vh)?,
            positions,
            query_gate: pointer(qg, c.head_dim)?,
            query_gate_stride: product(q, 2)? as u64,
            eps: c.rms_norm_eps,
            head_dim: u32_size(c.head_dim)?,
            rotary: u32_size(c.rotary_dim)?,
            qkv: 0..product(tokens, channels)?,
            z: zo..zo
                .checked_add(product(tokens, v)?)
                .ok_or_else(|| infer("GDN Z end overflow"))?,
            ab: abo..abo
                .checked_add(product(tokens, ab_width)?)
                .ok_or_else(|| infer("GDN AB end overflow"))?,
            qkv_width: channels,
            z_width: v,
            ab_width,
            batch: AttentionBatch {
                q: query,
                k: normalized_key,
                v: value,
                out: attended,
                q_offsets: cumulative,
                k_offsets: cumulative,
                q_tokens: tokens,
                k_tokens: tokens,
                batch: sequences,
                max_q: max_len,
                strides: [
                    stride(q)?,
                    c.head_dim as i64,
                    1,
                    stride(kv)?,
                    c.head_dim as i64,
                    1,
                    stride(kv)?,
                    c.head_dim as i64,
                    1,
                    stride(q)?,
                    c.head_dim as i64,
                    1,
                ],
            },
            block: None,
            norm,
            output,
            next_gammas,
        },
    })
}
