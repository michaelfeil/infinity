// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Ordered submission of one prepared Qwen3.5 Work onto the outer stream.
//! Every launch target is retained by the outer owners; this leaf allocates
//! nothing, binds no context, queries no device and owns no completion boundary.
use super::*;
use crate::gdn_prepared::{
    GdnBlockMetadata, GdnBlockPair, GdnBlockSuffix, GdnOwnedFrame, GdnStateViews,
};

/// The one GDN preparation shared by the ordinary fresh path and the B1 block
/// path. Both variants feed the same layer loop below.
enum PreparedGdn<'borrow, 'frame> {
    Fresh(&'borrow GdnOwnedFrame),
    Block(&'borrow GdnBlockPair<'frame>),
}

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
        if self.block.is_some() {
            return Err(infer(
                "Qwen3.5 block-prepared plan requires the block enqueue entry",
            ));
        }
        let frame = unsafe {
            gdn.prepare_owned_fresh_frame_from_device(
                self.gdn_inputs(slabs)?,
                cumulative,
                uploaded_cumulative,
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
        if self.block.is_none() {
            return Err(infer(
                "Qwen3.5 block entry requires a PreparedWork built with new_block",
            ));
        }
        if cumulative != [0, self.tokens].as_slice() {
            return Err(infer(
                "Qwen3.5 block entry requires exactly one B1 sequence boundary pair",
            ));
        }
        if layout.prefix_tokens() == 0 || layout.prefix_tokens() > layout.tokens() {
            return Err(infer(
                "Qwen3.5 block prefix must be nonzero and at most the token count",
            ));
        }
        if layout.tokens() != usize::try_from(self.tokens).map_err(infer)?
            || layout.value_heads() != usize::try_from(self.value_heads).map_err(infer)?
            || layout.channels() != self.qkv_width
            || layout.value_width()? != self.z_width
            || layout.ab_width()? != self.ab_width
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
        let prefix_inputs = self.block_gdn_inputs(slabs, layout, false)?;
        let suffix = if layout.suffix_launch_required() {
            Some(GdnBlockSuffix {
                inputs: self.block_gdn_inputs(slabs, layout, true)?,
                cumulative: layout.suffix_cumulative(),
                state: GdnStateViews::Cached {
                    initial: state,
                    history_rows: self.block_history(slabs, layout)?,
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
        // SAFETY: the first norm was prepared against these retained slabs.
        unsafe { self.first.enqueue(math, stream, self.eps)? };

        for (step, layer) in self.layers.iter().zip(weights.layers.iter()) {
            match (&step.attention, &layer.attention) {
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
                        q_norm.enqueue(math, stream, self.eps)?;
                        k_norm.enqueue(math, stream, self.eps)?;
                        math.rope_bound(
                            stream,
                            q_norm.out,
                            k_norm.out,
                            cosine,
                            sine,
                            self.positions,
                            self.tokens,
                            self.qheads,
                            self.kvheads,
                            64,
                            u64::from(self.qheads) * 256,
                            u64::from(self.kvheads) * 256,
                            table_stride,
                        )?;
                        if let Some(block) = &self.block {
                            block.enqueue_bound(fa4, stream)?;
                        } else {
                            fa4.forward_batch_bound(stream, &self.attention)?;
                        }
                        math.gate_bound(
                            stream,
                            self.attention.out,
                            self.query_gate,
                            self.attention.out,
                            self.tokens,
                            self.qheads,
                            self.query_gate_stride,
                        )?;
                        blas.linear_bound(output)?;
                    }
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
                    let inputs = self.gdn_inputs(slabs)?;
                    let mut joined_a = a.output;
                    let mut joined_b = b.output;
                    let mut joined_ab = inputs.ab.cu_deviceptr();
                    let mut joined_tokens = self.tokens;
                    let mut joined_heads = self.value_heads;
                    let pairs = product(
                        usize::try_from(self.tokens).map_err(infer)?,
                        product(2, usize::try_from(self.value_heads).map_err(infer)?)?,
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
                }
                _ => {
                    return Err(infer(
                        "Qwen3.5 prepared attention kind does not match retained layer weights",
                    ))
                }
            }

            // SAFETY: the post-attention norm and MLP plans retain their exact slabs.
            unsafe {
                step.post.enqueue(math, stream, self.eps)?;
                blas.linear_bound(&step.gate_up)?;
            }

            let mut silu_input = step.gate_up.output;
            let mut silu_output = step.down.x;
            let mut silu_rows = self.tokens;
            let mut silu_width = self.intermediate;
            let groups = usize::try_from(self.intermediate)
                .map_err(infer)?
                .div_ceil(8);
            let silu_grid = u32_size(
                product(usize::try_from(self.tokens).map_err(infer)?, groups)?.div_ceil(256),
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

            // SAFETY: down and next-residual plans bind retained MLP weights and slabs.
            unsafe {
                blas.linear_bound(&step.down)?;
                step.next.enqueue(math, stream, self.eps)?;
            }
        }
        Ok(())
    }
}
