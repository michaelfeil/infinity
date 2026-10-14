use crate::flash_attn::flash_attn_varlen;
use crate::layers::rotary::apply_packed_rotary;
use crate::layers::MlpLinear;
use crate::layers::{
    get_cos_sin, get_inv_freqs, index_select, CompactUnfoldTensors, HiddenAct, Linear, RMSNorm,
};
use crate::models::{Model, Qwen3Config};
use candle::{Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use infinity_v3_backend_core::{Batch, ModelType, Pool};

struct Qwen3Attention {
    qkv_proj: Linear,
    o_proj: Linear,

    q_norm: RMSNorm,
    k_norm: RMSNorm,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    attention_head_size: usize,

    softmax_scale: f32,
    use_bidirectional_attention: bool,

    span: tracing::Span,
}

impl Qwen3Attention {
    pub fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        if config.use_sliding_window {
            candle::bail!("Sliding window is not supported for Qwen3");
        }

        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let num_key_value_heads = config.num_key_value_heads;
        let hidden_size = config.hidden_size;

        let query_weight = vb.pp("q_proj").get(
            (num_attention_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let query_bias = if config.attention_bias {
            Some(
                vb.pp("q_proj")
                    .get(num_attention_heads * attention_head_size, "bias")?,
            )
        } else {
            None
        };

        let key_weight = vb.pp("k_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let key_bias = if config.attention_bias {
            Some(
                vb.pp("k_proj")
                    .get(num_key_value_heads * attention_head_size, "bias")?,
            )
        } else {
            None
        };

        let value_weight = vb.pp("v_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let value_bias = if config.attention_bias {
            Some(
                vb.pp("v_proj")
                    .get(num_key_value_heads * attention_head_size, "bias")?,
            )
        } else {
            None
        };
        let qkv_weight = Tensor::cat(&[query_weight, key_weight, value_weight], 0)?;
        let qkv_bias = match (query_bias, key_bias, value_bias) {
            (Some(q), Some(k), Some(v)) => Some(Tensor::cat(&[q, k, v], 0)?),
            _ => None,
        };
        let qkv_proj = Linear::new(qkv_weight, qkv_bias, None);

        let o_proj_weight = vb.pp("o_proj").get(
            (hidden_size, num_attention_heads * attention_head_size),
            "weight",
        )?;
        let o_proj_bias = if config.attention_bias {
            Some(vb.pp("o_proj").get(hidden_size, "bias")?)
        } else {
            None
        };
        let o_proj = Linear::new(o_proj_weight, o_proj_bias, None);

        let q_norm = RMSNorm::load(vb.pp("q_norm"), attention_head_size, config.rms_norm_eps)?;
        let k_norm = RMSNorm::load(vb.pp("k_norm"), attention_head_size, config.rms_norm_eps)?;

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            attention_head_size,
            softmax_scale,
            use_bidirectional_attention: config.use_bidirectional_attention,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
        compact_tensors: &CompactUnfoldTensors,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let qkv = self.qkv_proj.forward(hidden_states)?;
        let mut shape = hidden_states.dims().to_vec();
        shape.pop();
        shape.extend([
            self.num_attention_heads + 2 * self.num_key_value_heads,
            self.attention_head_size,
        ]);
        // Reshape before slicing: reshaping a strided view would copy Q/K.
        let qkv = qkv.reshape(shape)?;
        let q = qkv.narrow(candle::D::Minus2, 0, self.num_attention_heads)?;
        let k = qkv.narrow(
            candle::D::Minus2,
            self.num_attention_heads,
            self.num_key_value_heads,
        )?;
        let v = qkv.narrow(
            candle::D::Minus2,
            self.num_attention_heads + self.num_key_value_heads,
            self.num_key_value_heads,
        )?;
        #[cfg(feature = "cuda")]
        let fused =
            crate::layers::qk_norm_rope::try_forward(&q, &k, &self.q_norm, &self.k_norm, cos, sin)?;
        #[cfg(not(feature = "cuda"))]
        let fused: Option<(Tensor, Tensor)> = None;
        let (q, k) = match fused {
            Some(pair) => pair,
            None => {
                let (q, _) = self.q_norm.forward(&q.contiguous()?, None)?;
                let (k, _) = self.k_norm.forward(&k.contiguous()?, None)?;
                apply_packed_rotary(&q, &k, cos, sin)?
            }
        };

        // Expand Q, K, V to ORIGINAL layout for attention
        #[cfg(feature = "cuda")]
        let unfolded = compact_tensors
            .scatter_unfold
            .as_ref()
            .map(|ids| crate::layers::qk_norm_rope::try_unfold_qkv(&q, &k, &v, ids))
            .transpose()?
            .flatten();
        #[cfg(not(feature = "cuda"))]
        let unfolded: Option<(Tensor, Tensor, Tensor)> = None;
        let (q, k, v) = match unfolded {
            Some(qkv) => qkv,
            None => (
                compact_tensors.scatter_unfold(&q)?,
                compact_tensors.scatter_unfold(&k)?,
                compact_tensors.scatter_unfold(&v)?,
            ),
        };

        let attention = flash_attn_varlen(
            &q,
            &k,
            &v,
            None,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            self.softmax_scale,
            !self.use_bidirectional_attention,
            None,
            None,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        // Compact attention output back to COMPACT layout before o_proj
        let attention = compact_tensors.fold_gather(&attention)?;

        self.o_proj.forward(&attention)
    }
}

struct Qwen3MLP {
    gate_up_proj: MlpLinear,
    down_proj: MlpLinear,

    act: HiddenAct,

    span: tracing::Span,
}

impl Qwen3MLP {
    pub fn load(vb: VarBuilder, config: &Qwen3Config, enable_fp8_dynamic: bool) -> Result<Self> {
        let intermediate_size = config.intermediate_size;

        let gate_proj_weight = vb
            .pp("gate_proj")
            .get((intermediate_size, config.hidden_size), "weight")?;

        let up_proj_weight = vb
            .pp("up_proj")
            .get((intermediate_size, config.hidden_size), "weight")?;

        let gate_up_proj_weight = Tensor::cat(&[&gate_proj_weight, &up_proj_weight], 0)?;
        let gate_up_proj = MlpLinear::new(gate_up_proj_weight, enable_fp8_dynamic)?;

        let down_proj_weight = vb
            .pp("down_proj")
            .get((config.hidden_size, intermediate_size), "weight")?;
        let down_proj = MlpLinear::new(down_proj_weight, enable_fp8_dynamic)?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            act: config.hidden_act.clone(),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.gate_up_proj.forward(hidden_states)?;
        self.down_proj.forward_gated(&gate_up_states, &self.act)
    }
}

enum Qwen3FeedForward {
    Dense(Qwen3MLP),
    Moe(super::qwen3_moe::Qwen3Moe),
}
impl Qwen3FeedForward {
    fn load(
        vb: VarBuilder,
        config: &Qwen3Config,
        index: usize,
        enable_fp8_dynamic: bool,
    ) -> Result<Self> {
        if config.is_moe_layer(index)? {
            if enable_fp8_dynamic {
                candle::bail!("Dynamic FP8 is not supported for Qwen3 routed experts");
            }
            Ok(Self::Moe(super::qwen3_moe::Qwen3Moe::load(vb, config)?))
        } else {
            Ok(Self::Dense(Qwen3MLP::load(vb, config, enable_fp8_dynamic)?))
        }
    }
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden),
            Self::Moe(moe) => moe.forward(hidden),
        }
    }
}

struct Qwen3Layer {
    attention: Qwen3Attention,
    mlp: Qwen3FeedForward,
    input_layer_norm: RMSNorm,
    post_attention_layer_norm: RMSNorm,

    span: tracing::Span,
}

impl Qwen3Layer {
    pub fn load(
        vb: VarBuilder,
        config: &Qwen3Config,
        index: usize,
        enable_fp8_dynamic: bool,
    ) -> Result<Self> {
        let attention = Qwen3Attention::load(vb.pp("self_attn"), config)?;
        let mlp = Qwen3FeedForward::load(vb.pp("mlp"), config, index, enable_fp8_dynamic)?;

        let input_layer_norm = RMSNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let post_attention_layer_norm = RMSNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            attention,
            mlp,
            input_layer_norm,
            post_attention_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
        compact_tensors: &CompactUnfoldTensors,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        let (normed_hidden_states, res) = self.input_layer_norm.forward(hidden_states, residual)?;

        let attn_output = self.attention.forward(
            &normed_hidden_states,
            cu_seqlens,
            cos,
            sin,
            max_s,
            compact_tensors,
        )?;

        let (normed_attn_res_output, attn_res) = self
            .post_attention_layer_norm
            .forward(&attn_output, Some(&res))?;

        let mlp_output = self.mlp.forward(&normed_attn_res_output)?;

        Ok((mlp_output, attn_res))
    }
}

pub struct FlashQwen3Model {
    embeddings: Embedding,
    layers: Vec<Qwen3Layer>,
    norm: RMSNorm,
    linear_output_projection: Option<Linear>,
    cos_cache: Tensor,
    sin_cache: Tensor,
    pool: Pool,
    pub device: Device,
    use_bidirectional_attention: bool,

    span: tracing::Span,
}

impl FlashQwen3Model {
    pub fn load(
        vb: VarBuilder,
        config: &Qwen3Config,
        model_type: ModelType,
        enable_fp8_dynamic: bool,
    ) -> Result<Self> {
        crate::flash_attn::validate_packed_device(&vb)?;

        let pool = match model_type {
            ModelType::Decision => candle::bail!("Typed decisions require a Laya checkpoint"),
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Qwen3")
            }
            ModelType::Embedding(pool) => pool,
        };

        // The Qwen3-Reranker models contain the `model` key
        // https://huggingface.co/collections/Qwen/qwen3-reranker-6841b22d0192d7ade9cdefea
        let root = vb.clone();
        let vb = if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            vb
        };

        let embeddings = Embedding::new(
            vb.pp("embed_tokens")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );

        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                Qwen3Layer::load(
                    vb.pp(format!("layers.{index}")),
                    config,
                    index,
                    enable_fp8_dynamic,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        let linear_output_projection = config.load_output_projection(&root, &vb)?;

        let inv_freqs = get_inv_freqs(
            layers[0].attention.attention_head_size,
            config.rope_theta,
            vb.device(),
            None,
        )?;
        let (cos_cache, sin_cache) = get_cos_sin(
            config.max_position_embeddings,
            &inv_freqs,
            vb.dtype(),
            false,
        )?;

        Ok(Self {
            embeddings,
            layers,
            norm,
            linear_output_projection,
            cos_cache,
            sin_cache,
            pool,
            device: vb.device().clone(),
            use_bidirectional_attention: config.use_bidirectional_attention,
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    pub(super) fn multimodal_embeddings(
        &self,
        batch: &Batch,
        visual: Option<(&Tensor, &Tensor, &[Tensor])>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (_, compact) = CompactUnfoldTensors::from_batch(batch, &self.device)?;
        let ids = Tensor::new(batch.input_ids.as_slice(), &self.device)?;
        let mut hidden = self.embeddings.forward(&ids)?;
        if let Some((indices, images, _)) = visual {
            let scatter_indices = indices
                .unsqueeze(1)?
                .broadcast_as(images.shape())?
                .contiguous()?;
            hidden = hidden.scatter(&scatter_indices, images, 0)?;
        }
        let cu = Tensor::new(batch.cumulative_seq_lengths.as_slice(), &self.device)?;
        let mut residual = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let (h, r) = layer.forward(
                &hidden,
                residual.as_ref(),
                &cu,
                cos,
                sin,
                batch.max_length as usize,
                &compact,
            )?;
            if let Some((indices, _, deepstack)) = visual {
                if let Some(features) = deepstack.get(i) {
                    hidden = (h + r)?.index_add(indices, features, 0)?;
                    residual = None;
                    continue;
                }
            }
            hidden = h;
            residual = Some(r);
        }
        let (hidden, _) = self.norm.forward(&hidden, residual.as_ref())?;
        let indices: Vec<u32> = batch
            .pooled_indices
            .iter()
            .map(|&i| batch.cumulative_seq_lengths[i as usize + 1] - 1)
            .collect();
        index_select(&hidden, &Tensor::new(indices.as_slice(), &self.device)?, 0)
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.cumulative_seq_lengths.len() - 1;

        // Create compact/unfold tensors and get embeddings
        let (input_ids, compact_tensors) = CompactUnfoldTensors::from_batch(&batch, &self.device)?;
        let mut hidden_states = self.embeddings.forward(&input_ids)?.contiguous()?;

        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;
        #[cfg(feature = "fa4")]
        let _fa4_batch =
            crate::fa4_native::prepare_batch(&cu_seqlens, &batch.cumulative_seq_lengths)?;

        // sin and cos are applied on the compact formation, therefore should be on the compact array
        let cos = index_select(&self.cos_cache, &compact_tensors.position_ids_compact, 0)?;
        let sin = index_select(&self.sin_cache, &compact_tensors.position_ids_compact, 0)?;

        let mut residual = None;
        for layer in &self.layers {
            let (h, r) = layer.forward(
                &hidden_states,
                residual.as_ref(),
                &cu_seqlens,
                &cos,
                &sin,
                batch.max_length as usize,
                &compact_tensors,
            )?;
            hidden_states = h;
            residual = Some(r);
        }

        let (outputs, _) = self.norm.forward(&hidden_states, residual.as_ref())?;

        let outputs = if let Some(linear_output_projection) = &self.linear_output_projection {
            linear_output_projection.forward(&outputs)?
        } else {
            outputs
        };

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
            if batch_size > 1
                && (has_pooling_requests
                    || batch.raw_indices.iter().copied().ne(0..batch_size as u32))
            {
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

impl Model for FlashQwen3Model {
    fn supports_radix_mlp(&self) -> bool {
        !self.use_bidirectional_attention
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}

#[cfg(test)]
mod moe_radix_tests {
    use super::*;

    #[test]
    fn fused_qkv_matches_separate_gqa_projections() -> Result<()> {
        for bias in [false, true] {
            let config: Qwen3Config = serde_json::from_value(serde_json::json!({
                "attention_bias": bias, "vocab_size": 32, "hidden_size": 12,
                "head_dim": 8, "intermediate_size": 24, "num_hidden_layers": 1,
                "num_attention_heads": 4, "num_key_value_heads": 2, "hidden_act": "silu",
                "max_position_embeddings": 16, "rms_norm_eps": 0.000001,
                "rope_theta": 10000., "use_sliding_window": false, "eos_token_id": 2
            }))
            .map_err(candle::Error::wrap)?;
            let vars = candle_nn::VarMap::new();
            let vb = VarBuilder::from_varmap(&vars, candle::DType::F32, &Device::Cpu);
            Qwen3Attention::load(vb.clone(), &config)?;
            for (name, var) in vars.data().lock().unwrap().iter() {
                let offset = name.bytes().map(usize::from).sum::<usize>();
                let data = (0..var.elem_count())
                    .map(|i| ((i + offset) as f32).sin() * 0.1)
                    .collect::<Vec<_>>();
                var.set(&Tensor::from_vec(data, var.shape(), &Device::Cpu)?)?;
            }
            let model = Qwen3Attention::load(vb.clone(), &config)?;
            let input = Tensor::arange(0f32, 36f32, &Device::Cpu)?.reshape((3, 12))?;
            let mut separate = Vec::new();
            for (name, width) in [("q_proj", 32), ("k_proj", 16), ("v_proj", 16)] {
                let weight = vb.pp(name).get((width, 12), "weight")?;
                let bias = if bias {
                    Some(vb.pp(name).get(width, "bias")?)
                } else {
                    None
                };
                separate.push(Linear::new(weight, bias, None).forward(&input)?);
            }
            let error = (model.qkv_proj.forward(&input)? - Tensor::cat(&separate, 1)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(error < 1e-5, "QKV projection changed by {error}");
        }
        Ok(())
    }

    #[test]
    fn moe_radix_mlp_preserves_shared_prefix_outputs() -> Result<()> {
        let config: Qwen3Config = serde_json::from_value(serde_json::json!({
            "attention_bias": false, "vocab_size": 32, "hidden_size": 16,
            "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "hidden_act": "silu", "max_position_embeddings": 16,
            "rms_norm_eps": 0.000001, "rope_theta": 10000., "use_sliding_window": false,
            "eos_token_id": 2, "num_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 24, "decoder_sparse_step": 2
        }))
        .map_err(candle::Error::wrap)?;
        let vars = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, candle::DType::F32, &Device::Cpu);
        FlashQwen3Model::load(
            vb.clone(),
            &config,
            ModelType::Embedding(Pool::LastToken),
            false,
        )?;
        for (name, var) in vars.data().lock().unwrap().iter() {
            let offset: usize = name.bytes().map(usize::from).sum();
            let values: Vec<f32> = (0..var.elem_count())
                .map(|i| ((i + offset) as f32 * 0.13).sin() * 0.2)
                .collect();
            var.set(&Tensor::from_vec(values, var.shape(), &Device::Cpu)?)?;
        }
        let batch = Batch {
            multimodal: vec![],
            input_ids: vec![3, 4, 5, 3, 4, 6, 7, 3, 4, 5],
            token_type_ids: vec![0; 10],
            position_ids: vec![0, 1, 2, 0, 1, 2, 3, 0, 1, 2],
            cumulative_seq_lengths: vec![0, 3, 7, 10],
            max_length: 4,
            pooled_indices: vec![0, 1, 2],
            raw_indices: vec![0, 1, 2],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let mut folded = batch.clone();
        folded.compact_input_ids = Some(vec![3, 4, 5, 6, 7]);
        folded.compact_position_ids = Some(vec![0, 1, 2, 2, 3]);
        folded.scatter_unfold = Some(vec![0, 1, 2, 0, 1, 3, 4, 0, 1, 2]);
        folded.fold_gather = Some(vec![0, 1, 2, 5, 6]);
        for pool in [Pool::LastToken, Pool::Mean] {
            let model =
                FlashQwen3Model::load(vb.clone(), &config, ModelType::Embedding(pool), false)?;
            assert!(model.supports_radix_mlp());
            let plain = model.forward(batch.clone())?;
            let compact = model.forward(folded.clone())?;
            for (expected, actual) in [(plain.0, compact.0), (plain.1, compact.1)] {
                let expected = expected.unwrap();
                let actual = actual.unwrap();
                assert_eq!(expected.dims(), actual.dims());
                let error = (expected - actual)?.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(error < 1e-5, "RadixMLP changed output by {error}");
            }
        }
        let mut bidirectional = config.clone();
        bidirectional.use_bidirectional_attention = true;
        let model = FlashQwen3Model::load(
            vb,
            &bidirectional,
            ModelType::Embedding(Pool::LastToken),
            false,
        )?;
        assert!(!model.supports_radix_mlp());
        Ok(())
    }
}
