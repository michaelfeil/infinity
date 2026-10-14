use super::qwen35_config::{DecisionAttentionMode, Qwen35Config, Qwen35TextConfig, ReadoutConfig};
use crate::flash_attn::flash_attn_varlen;
use crate::layers::{get_cos_sin, get_inv_freqs, index_select, CompactUnfoldTensors, Linear};
use crate::models::{Model, Qwen3Config};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use infinity_v3_backend_core::{Batch, DecisionInput, DecisionOutput, ModelType, Pool};

struct Norm {
    weight: Tensor,
    eps: f64,
}
impl Norm {
    fn load(vb: VarBuilder, width: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get(width, "weight")?
                .to_dtype(DType::F32)?
                .affine(1., 1.)?,
            eps,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let f = x.to_dtype(DType::F32)?;
        let inv = (f.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?
            .sqrt()?
            .recip()?;
        f.broadcast_mul(&inv)?
            .broadcast_mul(&self.weight)?
            .to_dtype(x.dtype())
    }
}
fn linear(vb: VarBuilder, input: usize, output: usize) -> Result<Linear> {
    Ok(Linear::new(vb.get((output, input), "weight")?, None, None))
}
struct Dense {
    gate_up: Linear,
    down: Linear,
}
impl Dense {
    fn load(vb: VarBuilder, h: usize, i: usize) -> Result<Self> {
        Ok(Self {
            gate_up: Linear::new(
                Tensor::cat(
                    &[
                        vb.pp("gate_proj").get((i, h), "weight")?,
                        vb.pp("up_proj").get((i, h), "weight")?,
                    ],
                    0,
                )?,
                None,
                None,
            ),
            down: linear(vb.pp("down_proj"), i, h)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&crate::layers::gated_activation(
            &self.gate_up.forward(x)?,
            Some(&crate::layers::HiddenAct::Silu),
        )?)
    }
}
struct FullAttention {
    causal: bool,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    qn: Norm,
    kn: Norm,
    heads: usize,
    kv: usize,
    dim: usize,
    rotary: usize,
}
impl FullAttention {
    fn load(vb: VarBuilder, c: &Qwen35TextConfig) -> Result<Self> {
        let (h, n, k, d) = (
            c.hidden_size,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        );
        Ok(Self {
            causal: true,
            q: linear(vb.pp("q_proj"), h, 2 * n * d)?,
            k: linear(vb.pp("k_proj"), h, k * d)?,
            v: linear(vb.pp("v_proj"), h, k * d)?,
            o: linear(vb.pp("o_proj"), n * d, h)?,
            qn: Norm::load(vb.pp("q_norm"), d, c.rms_norm_eps)?,
            kn: Norm::load(vb.pp("k_norm"), d, c.rms_norm_eps)?,
            heads: n,
            kv: k,
            dim: d,
            rotary: c.rotary_dim(),
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        cu: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
        compact: &CompactUnfoldTensors,
    ) -> Result<Tensor> {
        let t = x.dim(0)?;
        let qg = self.q.forward(x)?.reshape((t, self.heads, 2 * self.dim))?;
        let q = self.qn.forward(&qg.narrow(2, 0, self.dim)?)?;
        let gate = qg
            .narrow(2, self.dim, self.dim)?
            .reshape((t, self.heads * self.dim))?;
        let k = self
            .kn
            .forward(&self.k.forward(x)?.reshape((t, self.kv, self.dim))?)?;
        let v = self.v.forward(x)?.reshape((t, self.kv, self.dim))?;
        let rotate = |x: &Tensor| -> Result<Tensor> {
            let r = x.narrow(2, 0, self.rotary)?;
            let half = self.rotary / 2;
            let other = Tensor::cat(&[r.narrow(2, half, half)?.neg()?, r.narrow(2, 0, half)?], 2)?;
            let rotated = (r.broadcast_mul(&cos.unsqueeze(1)?)?
                + other.broadcast_mul(&sin.unsqueeze(1)?)?)?;
            Tensor::cat(
                &[rotated, x.narrow(2, self.rotary, self.dim - self.rotary)?],
                2,
            )?
            .contiguous()
        };
        let q = compact.scatter_unfold(&rotate(&q)?)?;
        let k = compact.scatter_unfold(&rotate(&k)?)?;
        let v = compact.scatter_unfold(&v)?;
        let a = flash_attn_varlen(
            &q,
            &k,
            &v,
            None,
            cu,
            cu,
            max_s,
            max_s,
            (self.dim as f32).sqrt().recip(),
            self.causal,
            None,
            None,
        )?
        .flatten_from(1)?;
        let a = compact.fold_gather(&a)?;
        self.o.forward(&(a * candle_nn::ops::sigmoid(&gate)?)?)
    }
}
struct DeltaAttention {
    qkv: Linear,
    z: Linear,
    ab: Linear,
    out: Linear,
    delta: crate::layers::qwen35_gdn::GatedDelta,
}
impl DeltaAttention {
    fn load(vb: VarBuilder, c: &Qwen35TextConfig) -> Result<Self> {
        let (kh, vh, h) = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.hidden_size,
        );
        let channels = (2 * kh + vh) * 128;
        let ab = Tensor::cat(
            &[
                vb.pp("in_proj_a").get((vh, h), "weight")?,
                vb.pp("in_proj_b").get((vh, h), "weight")?,
            ],
            0,
        )?;
        Ok(Self {
            qkv: linear(vb.pp("in_proj_qkv"), h, channels)?,
            z: linear(vb.pp("in_proj_z"), h, vh * 128)?,
            ab: Linear::new(ab, None, None),
            out: linear(vb.pp("out_proj"), vh * 128, h)?,
            delta: crate::layers::qwen35_gdn::GatedDelta {
                conv: vb
                    .pp("conv1d")
                    .get((channels, 1, c.linear_conv_kernel_dim), "weight")?
                    .reshape((channels, c.linear_conv_kernel_dim))?,
                a_log: vb.to_dtype(DType::F32).get(vh, "A_log")?,
                dt_bias: vb.get(vh, "dt_bias")?.to_dtype(DType::F32)?,
                norm: vb.pp("norm").get(128, "weight")?,
                key_heads: kh,
                value_heads: vh,
                epsilon: c.rms_norm_eps as f32,
            },
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        lengths: &[u32],
        compact: &CompactUnfoldTensors,
    ) -> Result<Tensor> {
        let qkv = compact.scatter_unfold(&self.qkv.forward(x)?)?;
        let z = compact.scatter_unfold(&self.z.forward(x)?)?;
        let ab = compact.scatter_unfold(&self.ab.forward(x)?)?;
        self.out
            .forward(&compact.fold_gather(&self.delta.forward(&qkv, &z, &ab, lengths)?)?)
    }
}
enum Attention {
    Full(FullAttention),
    Linear(DeltaAttention),
}
enum FeedForward {
    Dense(Dense),
    Moe {
        experts: super::qwen3_moe::Qwen3Moe,
        shared: Dense,
        gate: Linear,
    },
}
impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(dense) => dense.forward(x),
            Self::Moe {
                experts,
                shared,
                gate,
            } => {
                experts.forward(x)?
                    + shared
                        .forward(x)?
                        .broadcast_mul(&candle_nn::ops::sigmoid(&gate.forward(x)?)?)?
            }
        }
    }
}
struct Layer {
    attn: Attention,
    input_norm: Norm,
    post_norm: Norm,
    mlp: FeedForward,
}
impl Layer {
    fn load(
        vb: VarBuilder,
        c: &Qwen35TextConfig,
        index: usize,
        moe_config: &Qwen3Config,
    ) -> Result<Self> {
        Ok(Self {
            attn: match c.layer_types[index].as_str() {
                "linear_attention" => {
                    Attention::Linear(DeltaAttention::load(vb.pp("linear_attn"), c)?)
                }
                "full_attention" => Attention::Full(FullAttention::load(vb.pp("self_attn"), c)?),
                _ => candle::bail!("Unknown Qwen3.5 attention layer"),
            },
            input_norm: Norm::load(vb.pp("input_layernorm"), c.hidden_size, c.rms_norm_eps)?,
            post_norm: Norm::load(
                vb.pp("post_attention_layernorm"),
                c.hidden_size,
                c.rms_norm_eps,
            )?,
            mlp: if c.num_experts == 0 {
                FeedForward::Dense(Dense::load(
                    vb.pp("mlp"),
                    c.hidden_size,
                    c.intermediate_size,
                )?)
            } else {
                FeedForward::Moe {
                    experts: super::qwen3_moe::Qwen3Moe::load(vb.pp("mlp"), moe_config)?,
                    shared: Dense::load(
                        vb.pp("mlp.shared_expert"),
                        c.hidden_size,
                        c.shared_expert_intermediate_size,
                    )?,
                    gate: linear(vb.pp("mlp.shared_expert_gate"), c.hidden_size, 1)?,
                }
            },
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        cu: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        batch: &Batch,
        compact: &CompactUnfoldTensors,
    ) -> Result<Tensor> {
        let normalized = self.input_norm.forward(x)?;
        let attention = match &self.attn {
            Attention::Full(a) => a.forward(
                &normalized,
                cu,
                cos,
                sin,
                batch.max_length as usize,
                compact,
            )?,
            Attention::Linear(a) => {
                a.forward(&normalized, &batch.cumulative_seq_lengths, compact)?
            }
        };
        let residual = (x + attention)?;
        let n = self.post_norm.forward(&residual)?;
        residual + self.mlp.forward(&n)?
    }
}
pub struct Qwen35Model {
    embeddings: Embedding,
    layers: Vec<Layer>,
    norm: Norm,
    linear_output_projection: Option<Linear>,
    decision_head: Option<Tensor>,
    readout: Option<Linear>,
    clef: Option<super::clef::JointSchemaHead>,
    vision: Option<super::qwen35_vision::Qwen35Vision>,
    cos_cache: Tensor,
    sin_cache: Tensor,
    pool: Pool,
    pub device: Device,
    use_bidirectional_attention: bool,
    span: tracing::Span,
}
impl Qwen35Model {
    pub fn load(
        vb: VarBuilder,
        config: &Qwen35Config,
        model_type: ModelType,
        external_readout: bool,
    ) -> Result<Self> {
        config.validate()?;
        let c = config.text();
        if !matches!(vb.device(), Device::Cuda(_)) || vb.dtype() != DType::BF16 {
            candle::bail!("Qwen3.5-MoE currently requires BF16 CUDA")
        }
        let decision = model_type == ModelType::Decision;
        let pool = match model_type {
            ModelType::Embedding(p) => p,
            ModelType::Decision => Pool::LastToken,
            _ => candle::bail!("Qwen3.5 requires embedding or decision mode"),
        };
        let model = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model.language_model")
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else if vb.contains_tensor("language_model.embed_tokens.weight") {
            vb.pp("language_model")
        } else {
            vb.clone()
        };
        let embeddings = Embedding::new(
            model
                .pp("embed_tokens")
                .get((c.vocab_size, c.hidden_size), "weight")?,
            c.hidden_size,
        );
        let decision_head = if decision && !external_readout {
            Some(if config.tie_word_embeddings() {
                embeddings.embeddings().clone()
            } else {
                vb.pp("lm_head")
                    .get((c.vocab_size, c.hidden_size), "weight")?
            })
        } else {
            None
        };
        let moe_config = c.moe_config()?;
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::load(model.pp(format!("layers.{i}")), c, i, &moe_config))
            .collect::<Result<Vec<_>>>()?;
        let norm = Norm::load(model.pp("norm"), c.hidden_size, c.rms_norm_eps)?;
        let inv = get_inv_freqs(
            c.rotary_dim(),
            c.rope_parameters.rope_theta,
            vb.device(),
            None,
        )?;
        let (cos_cache, sin_cache) =
            get_cos_sin(c.max_position_embeddings, &inv, vb.dtype(), true)?;
        Ok(Self {
            embeddings,
            layers,
            norm,
            linear_output_projection: None,
            decision_head,
            readout: None,
            clef: None,
            vision: None,
            cos_cache,
            sin_cache,
            pool,
            device: vb.device().clone(),
            use_bidirectional_attention: false,
            span: tracing::span!(tracing::Level::TRACE, "qwen35"),
        })
    }
    pub fn with_readout(mut self, path: &std::path::Path, config: &Qwen35Config) -> Result<Self> {
        let readout_config = ReadoutConfig::from_slice(
            &std::fs::read(path.join("decision_config.json")).map_err(candle::Error::wrap)?,
        )?;
        self.set_readout_attention_mode(readout_config.attention_mode);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[path.join("readout.safetensors")],
                DType::BF16,
                &self.device,
            )?
        };
        self.readout = Some(Linear::new(
            vb.get((255, config.text().hidden_size), "weight")?,
            None,
            None,
        ));
        Ok(self)
    }

    fn set_readout_attention_mode(&mut self, mode: DecisionAttentionMode) {
        self.use_bidirectional_attention = mode == DecisionAttentionMode::NoncausalFullAttention;
        for layer in &mut self.layers {
            if let Attention::Full(attention) = &mut layer.attn {
                attention.causal = !self.use_bidirectional_attention;
            }
        }
        // Linear attention retains its native causal convolution and recurrence.
    }

    pub fn with_clef(mut self, path: &std::path::Path, config: &Qwen35Config) -> Result<Self> {
        self.clef = Some(super::clef::JointSchemaHead::load(
            path,
            config.text().hidden_size,
            self.embeddings.embeddings().dtype(),
            &self.device,
        )?);
        Ok(self)
    }
    pub fn with_vision(mut self, vb: VarBuilder, config: &Qwen35Config) -> Result<Self> {
        if let Some(vision) = config.vision() {
            self.vision = Some(super::qwen35_vision::Qwen35Vision::load(
                vb,
                vision,
                config.text(),
            )?);
        }
        Ok(self)
    }
    fn forward_hidden(&self, batch: &Batch) -> Result<(Tensor, CompactUnfoldTensors)> {
        // Prefix states depend on each sequence's suffix after noncausal attention.
        // Reject folded batches even if a caller bypasses supports_radix_mlp().
        if self.use_bidirectional_attention
            && (batch.compact_input_ids.is_some()
                || batch.compact_position_ids.is_some()
                || batch.scatter_unfold.is_some()
                || batch.fold_gather.is_some())
        {
            candle::bail!("Noncausal Qwen3.5 attention does not support RadixMLP prefix folding")
        }
        let (ids, compact) = CompactUnfoldTensors::from_batch(batch, &self.device)?;
        let mut hidden = self.embeddings.forward(&ids)?.contiguous()?;
        let cu = Tensor::new(batch.cumulative_seq_lengths.as_slice(), &self.device)?;
        #[cfg(feature = "fa4")]
        let _fa4_batch = crate::fa4_native::prepare_batch(&cu, &batch.cumulative_seq_lengths)?;
        let has_images = batch
            .multimodal
            .iter()
            .flatten()
            .any(|media| !media.images.is_empty());
        let (cos, sin) = if has_images {
            if batch.compact_input_ids.is_some()
                && !batch
                    .scatter_unfold
                    .as_ref()
                    .zip(batch.fold_gather.as_ref())
                    .is_some_and(|(scatter, fold)| {
                        infinity_v3_backend_core::MultimodalEncoding::allows_radix_fold(
                            &batch.multimodal,
                            &batch.input_ids,
                            &batch.cumulative_seq_lengths,
                            scatter,
                            fold,
                        )
                    })
            {
                candle::bail!("Radix image folding requires identical images and image prefixes");
            }
            let vision = self
                .vision
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("Qwen3.5 vision weights unavailable".into()))?;
            let mut full_hidden = self
                .embeddings
                .forward(&Tensor::new(batch.input_ids.as_slice(), &self.device)?)?;
            vision.inject(batch, &mut full_hidden)?;
            hidden = compact.fold_gather(&full_hidden)?;
            let (cos, sin) = vision.rope(batch, &self.cos_cache, &self.sin_cache)?;
            (compact.fold_gather(&cos)?, compact.fold_gather(&sin)?)
        } else {
            (
                index_select(&self.cos_cache, &compact.position_ids_compact, 0)?,
                index_select(&self.sin_cache, &compact.position_ids_compact, 0)?,
            )
        };
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cu, &cos, &sin, batch, &compact)?;
        }
        Ok((self.norm.forward(&hidden)?, compact))
    }
    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let (outputs, compact_tensors) = self.forward_hidden(&batch)?;
        let batch_size = batch.cumulative_seq_lengths.len() - 1;

        let outputs = if let Some(linear_output_projection) = &self.linear_output_projection {
            linear_output_projection.forward(&outputs)?
        } else {
            outputs
        };

        self.pool_outputs(batch, outputs, compact_tensors, batch_size)
    }

    fn pool_outputs(
        &self,
        batch: Batch,
        outputs: Tensor,
        compact_tensors: CompactUnfoldTensors,
        batch_size: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;
        // Expand final outputs to original layout for pooling/raw extraction
        let outputs = compact_tensors.scatter_unfold(&outputs)?;
        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                // CLS and LastToken pooling
                Pool::Cls | Pool::LastToken => {
                    if batch_size > 1 {
                        // Get token indices form cu_seqlens
                        let mut indices = match self.pool {
                            Pool::Cls => cu_seqlens.narrow(0, 0, batch_size)?,
                            Pool::LastToken => {
                                let end = cu_seqlens.narrow(0, 1, batch_size)?;
                                (&end - &end.ones_like()?)?
                            }
                            _ => unreachable!(),
                        };

                        // If raw_indices is empty, we don't need to do anything with
                        // the pooled_indices
                        if has_raw_requests {
                            // We need the pooled indices to select the correct cls indices
                            let pooled_indices = Tensor::from_vec(
                                batch.pooled_indices.clone(),
                                batch.pooled_indices.len(),
                                &self.device,
                            )?;

                            // Only select indices that requires pooling
                            indices = index_select(&indices, &pooled_indices, 0)?
                        }

                        // Select tokens
                        Some(index_select(&outputs, &indices, 0)?)
                    } else {
                        Some(
                            match self.pool {
                                Pool::Cls => outputs.i(0)?,
                                Pool::LastToken => {
                                    outputs.i(batch.cumulative_seq_lengths[1] as usize - 1)?
                                }
                                _ => unreachable!(),
                            }
                            .unsqueeze(0)?,
                        )
                    }
                }
                // Mean pooling
                Pool::Mean => Some(crate::layers::mean_pool(
                    &outputs,
                    &batch.cumulative_seq_lengths,
                    &batch.pooled_indices,
                )?),
                Pool::Splade => {
                    unreachable!();
                }
            }
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                // Create indexing vector for the embeddings
                let shape = batch.input_ids.len();
                let mut final_indices: Vec<u32> = Vec::with_capacity(shape);
                for i in batch.raw_indices.into_iter() {
                    let i = i as usize;
                    // Get start/end token index of this specific member of the batch
                    let start = batch.cumulative_seq_lengths[i];
                    let end = batch.cumulative_seq_lengths[i + 1];

                    for j in start..end {
                        // Add indices for the tokens of this specific member of the batch
                        final_indices.push(j);
                    }
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

                // Select the tokens with final indices
                Some(index_select(&outputs, &final_indices, 0)?)
            } else {
                Some(outputs)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for Qwen35Model {
    fn decide(&self, batch: Batch, inputs: Vec<DecisionInput>) -> Result<Vec<DecisionOutput>> {
        if inputs.len() != batch.len() {
            candle::bail!("Decision metadata count does not match batch")
        }
        if let Some(readout) = &self.readout {
            let (states, compact) = self.forward_hidden(&batch)?;
            let states = compact.scatter_unfold(&states)?;
            let indices = Tensor::new(
                batch
                    .cumulative_seq_lengths
                    .iter()
                    .skip(1)
                    .map(|end| end - 1)
                    .collect::<Vec<_>>(),
                &self.device,
            )?;
            // Match upstream's BF16 readout, then promote logits for calibration.
            let logits = readout
                .forward(&index_select(&states, &indices, 0)?)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            return inputs
                .into_iter()
                .zip(logits)
                .map(|(input, logits)| {
                    let count = match input {
                        DecisionInput::OptionTokens { token_ids }
                            if !token_ids.is_empty()
                                && token_ids.iter().copied().eq(0..token_ids.len() as u32)
                                && token_ids.len() <= 255 =>
                        {
                            token_ids.len()
                        }
                        DecisionInput::Warmup => 1,
                        _ => candle::bail!("Pplx requires contiguous readout-row metadata"),
                    };
                    Ok(DecisionOutput {
                        memory: None,
                        logits: logits[..count].to_vec(),
                        action_probability: 1.0,
                    })
                })
                .collect();
        }
        let weights = self
            .decision_head
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("Qwen3.5 was not loaded for decisions".into()))?;
        if let Some(head) = &self.clef {
            let (hidden, compact) = self.forward_hidden(&batch)?;
            let hidden = if let Some(scatter) = &compact.scatter_unfold {
                index_select(&hidden, scatter, 0)?
            } else {
                hidden
            };
            return inputs
                .into_iter()
                .enumerate()
                .map(|(i, input)| {
                    let start = batch.cumulative_seq_lengths[i] as usize;
                    let end = batch.cumulative_seq_lengths[i + 1] as usize;
                    let fields = match input {
                        DecisionInput::Clef { fields } => fields,
                        DecisionInput::Warmup => vec![infinity_v3_backend_core::ClefField {
                            kind: 0,
                            question: (0, 1),
                            options: vec![(0, 1)],
                        }],
                        _ => candle::bail!("Expected Clef span metadata"),
                    };
                    let logits = head.forward(
                        &hidden.narrow(0, start, end - start)?,
                        &batch.input_ids[start..end],
                        &fields,
                        weights,
                    )?;
                    Ok(DecisionOutput {
                        memory: None,
                        logits,
                        action_probability: 1.0,
                    })
                })
                .collect();
        }
        let vocabulary = weights.dim(0)?;
        let tokens = inputs
            .into_iter()
            .map(|input| match input {
                DecisionInput::OptionTokens { token_ids }
                    if !token_ids.is_empty()
                        && token_ids.iter().all(|&id| (id as usize) < vocabulary) =>
                {
                    Ok(token_ids)
                }
                DecisionInput::Warmup => Ok(vec![0]),
                _ => candle::bail!("Qwen3.5 requires valid option-token metadata"),
            })
            .collect::<Result<Vec<_>>>()?;
        let (states, compact) = self.forward_hidden(&batch)?;
        let states = compact.scatter_unfold(&states)?;
        let indices = Tensor::from_vec(
            batch
                .cumulative_seq_lengths
                .iter()
                .skip(1)
                .map(|&end| end - 1)
                .collect::<Vec<_>>(),
            batch.len(),
            &self.device,
        )?;
        let last = index_select(&states, &indices, 0)?.to_dtype(DType::F32)?;
        tokens
            .into_iter()
            .enumerate()
            .map(|(i, ids)| {
                let selected = weights
                    .index_select(&Tensor::new(ids.as_slice(), &self.device)?, 0)?
                    .to_dtype(DType::F32)?;
                let logits = last.i(i)?.unsqueeze(0)?.matmul(&selected.t()?)?;
                Ok(DecisionOutput {
                    memory: None,
                    logits: logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?,
                    action_probability: 1.0,
                })
            })
            .collect()
    }

    fn supports_radix_mlp(&self) -> bool {
        !self.use_bidirectional_attention
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}

#[cfg(test)]
mod radix_tests {
    use super::*;

    fn batch(sequences: &[&[u32]], folded: bool) -> Batch {
        let mut batch = Batch {
            multimodal: vec![],
            input_ids: vec![],
            token_type_ids: vec![],
            position_ids: vec![],
            cumulative_seq_lengths: vec![0],
            max_length: 0,
            pooled_indices: (0..sequences.len() as u32).collect(),
            raw_indices: (0..sequences.len() as u32).collect(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let mut prefixes = std::collections::HashMap::new();
        let (mut ids, mut positions, mut scatter, mut gather) = (vec![], vec![], vec![], vec![]);
        for sequence in sequences {
            for (position, &token) in sequence.iter().enumerate() {
                let next = ids.len() as u32;
                let row = *prefixes
                    .entry(sequence[..=position].to_vec())
                    .or_insert_with(|| {
                        ids.push(token);
                        positions.push(position as u32);
                        gather.push(batch.input_ids.len() as u32);
                        next
                    });
                scatter.push(row);
                batch.input_ids.push(token);
                batch.position_ids.push(position as u32);
                batch.token_type_ids.push(0);
            }
            batch
                .cumulative_seq_lengths
                .push(batch.input_ids.len() as u32);
            batch.max_length = batch.max_length.max(sequence.len() as u32);
        }
        if folded {
            batch.compact_input_ids = Some(ids);
            batch.compact_position_ids = Some(positions);
            batch.scatter_unfold = Some(scatter);
            batch.fold_gather = Some(gather);
        }
        batch
    }

    fn close(expected: &Tensor, actual: &Tensor) -> Result<()> {
        assert_eq!(expected.dims(), actual.dims());
        let error = (expected.to_dtype(DType::F32)? - actual.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(error <= 0.02, "hybrid model changed by {error}");
        Ok(())
    }

    #[test]
    fn full_attention_respects_causality_and_sequence_boundaries() -> Result<()> {
        let device = Device::Cpu;
        let config: Qwen35TextConfig = serde_json::from_value(serde_json::json!({
            "hidden_size":16,"vocab_size":32,"num_hidden_layers":1,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":128,
            "max_position_embeddings":32,"rms_norm_eps":1e-6,
            "layer_types":["full_attention"],"intermediate_size":32,
            "rope_parameters":{"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":0.5},
            "linear_num_key_heads":1,"linear_num_value_heads":2,
            "linear_key_head_dim":128,"linear_value_head_dim":128,
            "linear_conv_kernel_dim":4,"hidden_act":"silu"
        })).map_err(candle::Error::wrap)?;
        let vars = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
        FullAttention::load(vb.clone(), &config)?;
        for (name, var) in vars.data().lock().unwrap().iter() {
            let offset: usize = name.bytes().map(usize::from).sum();
            let values: Vec<f32> = (0..var.elem_count())
                .map(|i| ((i + offset) as f32 * 0.13).sin() * 0.1)
                .collect();
            var.set(&Tensor::from_vec(values, var.shape(), &device)?)?;
        }
        let mut attention = FullAttention::load(vb, &config)?;
        let a: &[u32] = &[3, 4, 5, 6];
        let b: &[u32] = &[3, 4, 9, 10, 11];
        let forward = |attention: &FullAttention, sequences: &[&[u32]]| -> Result<Tensor> {
            let batch = batch(sequences, false);
            let (_, compact) = CompactUnfoldTensors::from_batch(&batch, &device)?;
            let values: Vec<f32> = batch
                .input_ids
                .iter()
                .flat_map(|&id| (0..16).map(move |i| ((id as usize * 16 + i) as f32 * 0.17).sin()))
                .collect();
            let count = batch.input_ids.len();
            attention.forward(
                &Tensor::from_vec(values, (count, 16), &device)?,
                &Tensor::new(batch.cumulative_seq_lengths, &device)?,
                &Tensor::ones((count, config.rotary_dim()), DType::F32, &device)?,
                &Tensor::zeros((count, config.rotary_dim()), DType::F32, &device)?,
                batch.max_length as usize,
                &compact,
            )
        };
        let error = |a: &Tensor, b: &Tensor| -> Result<f32> {
            (a - b)?.abs()?.max_all()?.to_scalar::<f32>()
        };
        let causal_a = forward(&attention, &[a])?;
        let causal_b = forward(&attention, &[b])?;
        assert!(error(&causal_a.narrow(0, 0, 2)?, &causal_b.narrow(0, 0, 2)?)? < 1e-6);
        attention.causal = false;
        let single_a = forward(&attention, &[a])?;
        let single_b = forward(&attention, &[b])?;
        assert!(error(&single_a.narrow(0, 0, 2)?, &single_b.narrow(0, 0, 2)?)? > 1e-5);
        let packed = forward(&attention, &[a, b])?;
        assert!(error(&packed.narrow(0, 0, a.len())?, &single_a)? < 1e-6);
        assert!(error(&packed.narrow(0, a.len(), b.len())?, &single_b)? < 1e-6);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA BF16 and FlashAttention"]
    fn hybrid_moe_preserves_prefix_folding_and_request_isolation() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config: Qwen35Config = serde_json::from_value(serde_json::json!({
            "hidden_size":32,"vocab_size":32,"num_hidden_layers":3,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":128,
            "max_position_embeddings":32,"rms_norm_eps":1e-6,
            "layer_types":["linear_attention","full_attention","linear_attention"],
            "rope_parameters":{"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":0.5},
            "num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":32,
            "shared_expert_intermediate_size":32,"linear_num_key_heads":1,
            "linear_num_value_heads":2,"linear_key_head_dim":128,"linear_value_head_dim":128,
            "linear_conv_kernel_dim":4,"hidden_act":"silu"
        })).map_err(candle::Error::wrap)?;
        let vars = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::BF16, &device);
        Qwen35Model::load(
            vb.clone(),
            &config,
            ModelType::Embedding(Pool::LastToken),
            false,
        )?;
        for (name, var) in vars.data().lock().unwrap().iter() {
            let offset: usize = name.bytes().map(usize::from).sum();
            let values: Vec<f32> = (0..var.elem_count())
                .map(|i| ((i + offset) as f32 * 0.13).sin() * 0.1)
                .collect();
            var.set(&Tensor::from_vec(values, var.shape(), &device)?.to_dtype(var.dtype())?)?;
        }
        let a: &[u32] = &[3, 4, 5, 6, 7, 8];
        let b: &[u32] = &[3, 4, 5, 6, 9, 10, 11];
        for pool in [Pool::LastToken, Pool::Mean] {
            let model = Qwen35Model::load(vb.clone(), &config, ModelType::Embedding(pool), false)?;
            assert!(model.supports_radix_mlp());
            for sequences in [[a, b, a], [b, a, b]] {
                let expected = model.forward(batch(&sequences, false))?;
                for _ in 0..2 {
                    let actual = model.forward(batch(&sequences, true))?;
                    close(expected.0.as_ref().unwrap(), actual.0.as_ref().unwrap())?;
                    close(expected.1.as_ref().unwrap(), actual.1.as_ref().unwrap())?;
                }
                let mut offset = 0;
                for (index, sequence) in sequences.iter().enumerate() {
                    let single = model.forward(batch(&[*sequence], false))?;
                    close(
                        &expected.0.as_ref().unwrap().narrow(0, index, 1)?,
                        single.0.as_ref().unwrap(),
                    )?;
                    close(
                        &expected
                            .1
                            .as_ref()
                            .unwrap()
                            .narrow(0, offset, sequence.len())?,
                        single.1.as_ref().unwrap(),
                    )?;
                    offset += sequence.len();
                }
            }
        }
        let mut model =
            Qwen35Model::load(vb, &config, ModelType::Embedding(Pool::LastToken), false)?;
        let causal_a = model.forward(batch(&[a], false))?.1.unwrap();
        let causal_b = model.forward(batch(&[b], false))?.1.unwrap();
        close(&causal_a.narrow(0, 0, 4)?, &causal_b.narrow(0, 0, 4)?)?;
        let saved = ReadoutConfig::from_slice(
            br#"{"attention_mode":"noncausal_full_attention","pooling":"last"}"#,
        )?;
        model.set_readout_attention_mode(saved.attention_mode);
        assert!(!model.supports_radix_mlp());
        assert!(model.forward(batch(&[a, b], true)).is_err());
        let packed = model.forward(batch(&[a, b], false))?.1.unwrap();
        let single_a = model.forward(batch(&[a], false))?.1.unwrap();
        let single_b = model.forward(batch(&[b], false))?.1.unwrap();
        close(&packed.narrow(0, 0, a.len())?, &single_a)?;
        close(&packed.narrow(0, a.len(), b.len())?, &single_b)?;
        let prefix_change = (single_a.narrow(0, 0, 4)?.to_dtype(DType::F32)?
            - single_b.narrow(0, 0, 4)?.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
        assert!(
            prefix_change > 1e-4,
            "noncausal prefix ignored its suffix: {prefix_change}"
        );
        Ok(())
    }
}
