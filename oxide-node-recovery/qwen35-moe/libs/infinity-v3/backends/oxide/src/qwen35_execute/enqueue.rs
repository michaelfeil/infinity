// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Ordered submission of one prepared Qwen3.5 Work onto the outer stream.
//! Every launch target is retained by the outer owners; this leaf allocates
//! nothing, binds no context, queries no device and owns no completion boundary.
use super::*;
use crate::gdn_prepared::{GdnBlockMetadata, GdnBlockSuffix, GdnStateViews};

impl PreparedWork {
    /// Enqueue the complete prepared text backbone on an already-bound stream.
    ///
    /// # Safety
    /// The enclosing Work MUST already have bound the device context used by
    /// `weights`, `blas`, `math`, `aux`, `fa4`, `stream` and `gdn`. It MUST keep
    /// every weight, slab, table, GDN allocation, borrowed per-layer weight
    /// binding, and the immutable `cumulative` metadata alive and unmodified
    /// until `stream` drains. No caller may reuse an input or slab except by the
    /// same-stream overwrite order encoded below. On ANY error the caller MUST
    /// drain or quarantine the whole stream/work, including every layer and the
    /// prepared GDN frame already queued here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &self,
        weights: &Qwen35DeviceWeights,
        blas: &Blas,
        math: &Qwen35Kernels,
        aux: &Qwen35AuxKernels,
        fa4: &Fa4,
        stream: &Stream,
        slabs: &Slabs<'_>,
        gdn: &mut GdnPrepared,
        cumulative: &[u32],
        uploaded_cumulative: GdnBufferView<'_, u32>,
        cosine: u64,
        sine: u64,
        table_stride: u32,
    ) -> Result<(), BackendError> {
        if self.attention.frame.block.is_some() {
            return Err(infer(
                "Qwen3.5 block-prepared plan requires the block enqueue entry",
            ));
        }
        let frame = unsafe {
            self.attention
                .prepare_fresh_frame(gdn, slabs, cumulative, uploaded_cumulative)?
        };
        unsafe {
            self.enqueue_prepared(
                weights,
                blas,
                math,
                aux,
                fa4,
                stream,
                slabs,
                gdn,
                &PreparedGdn::Fresh(&frame),
                cosine,
                sine,
                table_stride,
            )
        }
    }

    /// Enqueue a prepared B1 block request. The prefix and optional cached
    /// suffix GDN pair is admitted once here, before the single layer loop.
    ///
    /// # Safety
    /// The same outer Work safety contract as [`Self::enqueue`] applies. In
    /// addition, `input`, `prefix_state`, `layout` and every borrowed slab must
    /// stay alive and unmodified until `stream` drains; `layout` must describe
    /// the metadata already uploaded at `metadata_base` on this stream. No host
    /// transfer, allocation, query or synchronize occurs in this call.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // Consumed by the unadmitted Qwen3.5 runtime owner.
    pub(crate) unsafe fn enqueue_block<'a>(
        &self,
        weights: &Qwen35DeviceWeights,
        blas: &Blas,
        math: &Qwen35Kernels,
        aux: &Qwen35AuxKernels,
        fa4: &Fa4,
        stream: &Stream,
        slabs: &'a Slabs<'a>,
        gdn: &mut GdnPrepared,
        cumulative: &[u32],
        uploaded_cumulative: GdnBufferView<'_, u32>,
        cosine: u64,
        sine: u64,
        table_stride: u32,
        input: &'a DeviceBuffer<u32>,
        prefix_state: &'a DeviceBuffer<f32>,
        layout: &'a Qwen35BlockLayout,
        metadata_base: usize,
        combined_output: Range<usize>,
    ) -> Result<(), BackendError> {
        let _ = uploaded_cumulative;
        let attention = &self.attention;
        if attention.frame.block.is_none() {
            return Err(infer(
                "Qwen3.5 block entry requires a PreparedWork built with new_block",
            ));
        }
        if cumulative != [0, attention.frame.tokens].as_slice() {
            return Err(infer(
                "Qwen3.5 block entry requires exactly one B1 sequence boundary pair",
            ));
        }
        if layout.prefix_tokens() == 0 || layout.prefix_tokens() > layout.tokens() {
            return Err(infer(
                "Qwen3.5 block prefix must be nonzero and at most the token count",
            ));
        }
        if layout.tokens() != usize::try_from(attention.frame.tokens).map_err(infer)?
            || layout.value_heads()
                != usize::try_from(attention.frame.value_heads).map_err(infer)?
            || layout.channels() != attention.frame.qkv_width
            || layout.value_width()? != attention.frame.z_width
            || layout.ab_width()? != attention.frame.ab_width
        {
            return Err(infer(
                "Qwen3.5 block layout does not match the prepared frame geometry",
            ));
        }
        let (prefix_rows, suffix_rows) =
            layout.split_rows(layout.value_width()?, "combined output rows")?;
        let output_end = combined_output
            .start
            .checked_add(suffix_rows.end)
            .ok_or_else(|| infer("Qwen3.5 combined output range overflow"))?;
        if output_end != combined_output.end {
            return Err(infer(
                "Qwen3.5 combined output range must cover every value row",
            ));
        }
        let prefix_output = GdnBufferView::from_tensor_buffer_range(
            slabs.activation,
            shift(prefix_rows, combined_output.start, "block prefix output")?,
        )?;
        let suffix_output = shift(suffix_rows, combined_output.start, "block suffix output")?;
        let state =
            GdnBufferView::from_buffer_range(prefix_state, 0..layout.prefix_state_elements())?;
        let metadata = GdnBlockMetadata {
            prefix_cu: GdnBufferView::from_buffer_range(
                input,
                layout.prefix_cu_range(metadata_base)?,
            )?,
            suffix_cu: GdnBufferView::from_buffer_range(
                input,
                layout.suffix_cu_range(metadata_base)?,
            )?,
            history_cu: GdnBufferView::from_buffer_range(
                input,
                layout.history_cu_range(metadata_base)?,
            )?,
            sources: GdnBufferView::from_buffer_range(input, layout.source_range(metadata_base)?)?,
        };
        let prefix_inputs = attention.block_gdn_inputs(slabs, layout, false)?;
        let suffix = if layout.suffix_launch_required() {
            Some(GdnBlockSuffix {
                inputs: attention.block_gdn_inputs(slabs, layout, true)?,
                cumulative: layout.suffix_cumulative(),
                state: GdnStateViews::Cached {
                    initial: state,
                    history_rows: attention.block_history(slabs, layout)?,
                    history_cumulative: layout.history_cumulative(),
                    sources: layout.sources(),
                },
                final_state: None,
                output: GdnBufferView::from_tensor_buffer_range(slabs.activation, suffix_output)?,
            })
        } else {
            None
        };
        let pair = unsafe {
            gdn.prepare_block_pair(
                prefix_inputs,
                layout.prefix_cumulative(),
                suffix,
                metadata,
                state,
                prefix_output,
            )
        }
        .map_err(|failure| failure.error)?;
        unsafe {
            self.enqueue_prepared(
                weights,
                blas,
                math,
                aux,
                fa4,
                stream,
                slabs,
                gdn,
                &PreparedGdn::Block(&pair),
                cosine,
                sine,
                table_stride,
            )
        }
    }

    /// One layer loop shared by the ordinary fresh and B1 block GDN choices.
    #[allow(clippy::too_many_arguments)]
    unsafe fn enqueue_prepared<'a>(
        &self,
        weights: &Qwen35DeviceWeights,
        blas: &Blas,
        math: &Qwen35Kernels,
        aux: &Qwen35AuxKernels,
        fa4: &Fa4,
        stream: &Stream,
        slabs: &Slabs<'a>,
        gdn: &mut GdnPrepared,
        prepared: &PreparedGdn<'_, 'a>,
        cosine: u64,
        sine: u64,
        table_stride: u32,
    ) -> Result<(), BackendError> {
        if self.ffn.len() != weights.layers.len() || self.next.len() != weights.layers.len() {
            return Err(infer(
                "Qwen3.5 prepared dense FFN count differs from retained layers",
            ));
        }
        // SAFETY: the first norm was prepared against these retained slabs.
        unsafe { self.attention.enqueue_first(math, stream)? };

        for (index, ((step, next), layer)) in self
            .ffn
            .iter()
            .zip(self.next.iter())
            .zip(weights.layers.iter())
            .enumerate()
        {
            // SAFETY: prepared attention plans bind retained weights and slabs.
            unsafe {
                self.attention.enqueue_attention(
                    index,
                    &layer.attention,
                    blas,
                    math,
                    aux,
                    fa4,
                    stream,
                    slabs,
                    gdn,
                    prepared,
                    cosine,
                    sine,
                    table_stride,
                )?;
                self.attention.enqueue_post(index, math, stream)?;
                blas.linear_bound(&step.gate_up)?;
            }

            let mut silu_input = step.gate_up.output;
            let mut silu_output = step.down.x;
            let mut silu_rows = self.attention.frame.tokens;
            let mut silu_width = self.intermediate;
            let groups = usize::try_from(self.intermediate)
                .map_err(infer)?
                .div_ceil(8);
            let silu_grid = u32_size(
                product(
                    usize::try_from(self.attention.frame.tokens).map_err(infer)?,
                    groups,
                )?
                .div_ceil(256),
            )?;
            launch(
                &aux.silu,
                stream,
                (silu_grid, 1, 1),
                0,
                &mut [
                    arg(&mut silu_input),
                    arg(&mut silu_output),
                    arg(&mut silu_rows),
                    arg(&mut silu_width),
                ],
            )?;

            // SAFETY: down and next-residual plans bind retained weights and slabs.
            unsafe {
                blas.linear_bound(&step.down)?;
                next.enqueue(math, stream, self.attention.frame.eps)?;
            }
        }
        Ok(())
    }
}

/// Enqueue just the attention step for one prepared layer. This is the single
/// implementation shared by the dense donor and the Qwen3.5-MoE full loop.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn enqueue_attention_layer(
    work: &PreparedAttentionWork,
    index: usize,
    layer: &Qwen35DeviceAttention,
    blas: &Blas,
    math: &Qwen35Kernels,
    aux: &Qwen35AuxKernels,
    fa4: &Fa4,
    stream: &Stream,
    slabs: &Slabs<'_>,
    gdn: &mut GdnPrepared,
    prepared: &PreparedGdn<'_, '_>,
    cosine: u64,
    sine: u64,
    table_stride: u32,
) -> Result<(), BackendError> {
    let step = work
        .layers
        .get(index)
        .ok_or_else(|| infer("Qwen3.5 attention layer index is out of range"))?;
    let frame = &work.frame;
    match (&step.attention, layer) {
        (
            AttentionStep::Full {
                qg,
                key,
                value,
                output,
                q_norm,
                k_norm,
            },
            Qwen35DeviceAttention::Full { .. },
        ) => {
            // SAFETY: prepared full-attention plans bind retained weights and slabs.
            unsafe {
                blas.linear_bound(qg)?;
                blas.linear_bound(key)?;
                blas.linear_bound(value)?;
                q_norm.enqueue(math, stream, frame.eps)?;
                k_norm.enqueue(math, stream, frame.eps)?;
                math.rope_bound(
                    stream,
                    q_norm.out,
                    k_norm.out,
                    cosine,
                    sine,
                    frame.positions,
                    frame.tokens,
                    frame.qheads,
                    frame.kvheads,
                    frame.rotary,
                    u64::from(frame.qheads) * u64::from(frame.head_dim),
                    u64::from(frame.kvheads) * u64::from(frame.head_dim),
                    table_stride,
                )?;
                if let Some(block) = &frame.block {
                    block.enqueue_bound(fa4, stream)?;
                } else {
                    fa4.forward_batch_bound(stream, &frame.batch)?;
                }
                math.gate_bound(
                    stream,
                    frame.batch.out,
                    frame.query_gate,
                    frame.batch.out,
                    frame.tokens,
                    frame.qheads,
                    frame.query_gate_stride,
                )?;
                blas.linear_bound(output)?;
            }
            Ok(())
        }
        (
            AttentionStep::Linear {
                qkv,
                z,
                a,
                b,
                output,
            },
            Qwen35DeviceAttention::Linear {
                weights: layer_weights,
                ..
            },
        ) => {
            let inputs = work.gdn_inputs(slabs)?;
            let mut joined_a = a.output;
            let mut joined_b = b.output;
            let mut joined_ab = inputs.ab.cu_deviceptr();
            let mut joined_tokens = frame.tokens;
            let mut joined_heads = frame.value_heads;
            let pairs = product(
                usize::try_from(frame.tokens).map_err(infer)?,
                product(2, usize::try_from(frame.value_heads).map_err(infer)?)?,
            )?;
            let join_grid = u32_size(pairs.div_ceil(256))?;
            // SAFETY: the four prepared linear plans and join spans are retained.
            unsafe {
                blas.linear_bound(qkv)?;
                blas.linear_bound(z)?;
                blas.linear_bound(a)?;
                blas.linear_bound(b)?;
                launch(
                    &aux.join,
                    stream,
                    (join_grid, 1, 1),
                    0,
                    &mut [
                        arg(&mut joined_a),
                        arg(&mut joined_b),
                        arg(&mut joined_ab),
                        arg(&mut joined_tokens),
                        arg(&mut joined_heads),
                    ],
                )?;
                let projected_input = match prepared {
                    PreparedGdn::Fresh(frame) => gdn
                        .enqueue_owned_in_frame(
                            frame,
                            inputs,
                            GdnState::Fresh,
                            layer_weights.as_ref(),
                            false,
                        )
                        .map_err(|failure| failure.error)?
                        .output
                        .cu_deviceptr(),
                    PreparedGdn::Block(pair) => {
                        let prefix = gdn
                            .enqueue_in_block_frame(pair.prefix, layer_weights.as_ref())
                            .map_err(|failure| failure.error)?;
                        let base = prefix.output.cu_deviceptr();
                        if let Some(suffix) = pair.suffix {
                            let _ = gdn
                                .enqueue_in_block_frame(suffix, layer_weights.as_ref())
                                .map_err(|failure| failure.error)?;
                        }
                        base
                    }
                };
                let mut projected = *output;
                projected.x = projected_input;
                blas.linear_bound(&projected)?;
            }
            Ok(())
        }
        _ => Err(infer(
            "Qwen3.5 prepared attention kind does not match retained layer weights",
        )),
    }
}
