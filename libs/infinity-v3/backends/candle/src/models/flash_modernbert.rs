use std::collections::HashMap;

use crate::flash_attn::flash_attn_varlen;
use crate::layers::rotary::apply_packed_rotary;
use crate::layers::{get_cos_sin, get_inv_freqs, index_select, LayerNormNoBias, Linear};
use crate::models::modernbert::{
    ClassificationHead, ModernBertClassificationHead, ModernBertConfig, ModernBertEmbeddings,
    ModernBertMLP,
};
use crate::models::Model;
use candle::{Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use infinity_v3_backend_core::{Batch, ModelType, Pool};

struct ModernBertAttention {
    wqkv: Linear,
    wo: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f32,
    local_attention: usize,
    use_local_attention: bool,

    span: tracing::Span,
}

impl ModernBertAttention {
    pub fn load(vb: VarBuilder, index: usize, config: &ModernBertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let hidden_size = config.hidden_size;

        let wqkv_weight = vb
            .pp("Wqkv")
            .get((hidden_size * 3, hidden_size), "weight")?;
        let wqkv_bias = if config.attention_bias {
            vb.pp("Wqkv").get(hidden_size * 3, "bias").ok()
        } else {
            None
        };
        let wqkv: Linear = Linear::new(wqkv_weight, wqkv_bias, None);

        let wo_weight = vb.pp("Wo").get((hidden_size, hidden_size), "weight")?;
        let wo_bias = if config.attention_bias {
            vb.pp("Wo").get(hidden_size, "bias").ok()
        } else {
            None
        };
        let wo = Linear::new(wo_weight, wo_bias, None);

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

        let use_local_attention = !index.is_multiple_of(config.global_attn_every_n_layers);

        Ok(Self {
            wqkv,
            wo,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            softmax_scale,
            local_attention: config.local_attention / 2,
            use_local_attention,
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
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let qkv = self.wqkv.forward(hidden_states)?;

        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads * 3);
        new_qkv_shape.push(self.attention_head_size);
        let qkv = qkv.reshape(new_qkv_shape.as_slice())?;

        // Split qkv tensor
        let q = qkv.narrow(1, 0, self.num_attention_heads)?;
        let k = qkv.narrow(1, self.num_attention_heads, self.num_attention_heads)?;
        let v = qkv.narrow(1, self.num_attention_heads * 2, self.num_attention_heads)?;

        let (q, k) = apply_packed_rotary(&q, &k, cos, sin)?;

        let window_size = if self.use_local_attention {
            Some(self.local_attention)
        } else {
            None
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
            false,
            window_size,
            window_size,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        let hidden_states = self.wo.forward(&attention)?;

        Ok(hidden_states)
    }
}

struct ModernBertEncoderLayer {
    attn_norm: Option<LayerNormNoBias>,
    attn: ModernBertAttention,
    mlp_norm: LayerNormNoBias,
    mlp: ModernBertMLP,

    span: tracing::Span,
}

impl ModernBertEncoderLayer {
    pub fn load(
        vb: VarBuilder,
        index: usize,
        config: &ModernBertConfig,
        enable_fp8_dynamic: bool,
    ) -> Result<Self> {
        let attn_norm = if index != 0 {
            Some(LayerNormNoBias::load(
                vb.pp("attn_norm"),
                config.hidden_size,
                config.norm_eps as f32,
            )?)
        } else {
            None
        };

        let attn = ModernBertAttention::load(vb.pp("attn"), index, config)?;

        let mlp_norm = LayerNormNoBias::load(
            vb.pp("mlp_norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )?;
        let mlp = ModernBertMLP::load(vb.pp("mlp"), config, enable_fp8_dynamic)?;

        let span = tracing::span!(tracing::Level::TRACE, "layer");

        Ok(ModernBertEncoderLayer {
            attn_norm,
            attn,
            mlp_norm,
            mlp,
            span,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        // Carry the MLP output and residual separately between layers so the
        // next normalization can perform the rounded addition in its kernel.
        let (attn_norm, residual) = match (&self.attn_norm, residual) {
            (Some(norm), Some(residual)) => {
                norm.forward_rounded_residual(hidden_states, residual)?
            }
            (Some(norm), None) => (norm.forward(hidden_states, None)?, hidden_states.clone()),
            (None, None) => (hidden_states.clone(), hidden_states.clone()),
            (None, Some(_)) => {
                candle::bail!("ModernBERT first layer cannot have a pending residual")
            }
        };
        let attn_outputs = self.attn.forward(&attn_norm, cu_seqlens, cos, sin, max_s)?;
        let (normed, residual) = self
            .mlp_norm
            .forward_rounded_residual(&attn_outputs, &residual)?;
        Ok((self.mlp.forward(&normed)?, residual))
    }
}

struct ModernBertEncoder {
    layers: Vec<ModernBertEncoderLayer>,

    global_attn_every_n_layers: usize,

    span: tracing::Span,
}

impl ModernBertEncoder {
    pub fn load(
        vb: VarBuilder,
        config: &ModernBertConfig,
        enable_fp8_dynamic: bool,
    ) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                ModernBertEncoderLayer::load(
                    vb.pp(format!("{index}")),
                    index,
                    config,
                    enable_fp8_dynamic,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(ModernBertEncoder {
            layers,
            global_attn_every_n_layers: config.global_attn_every_n_layers,
            span,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
        rotary_cache: &HashMap<bool, (Tensor, Tensor)>,
        max_s: usize,
        final_norm: &LayerNormNoBias,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();
        let mut residual = None;

        for (index, layer) in self.layers.iter().enumerate() {
            let use_local_attention = index % self.global_attn_every_n_layers != 0;
            let (cos, sin) = &rotary_cache[&use_local_attention];

            let (output, next_residual) = layer.forward(
                &hidden_states,
                residual.as_ref(),
                cu_seqlens,
                cos,
                sin,
                max_s,
            )?;
            hidden_states = output;
            residual = Some(next_residual);
        }

        match residual {
            Some(residual) => {
                final_norm.forward_rounded_residual_norm_only(&hidden_states, &residual)
            }
            None => final_norm.forward(&hidden_states, None),
        }
    }
}

pub struct FlashModernBertModel {
    embeddings: ModernBertEmbeddings,
    encoder: ModernBertEncoder,
    final_norm: LayerNormNoBias,
    pool: Pool,
    classifier: Option<Box<dyn ClassificationHead + Send>>,

    rotary_cache: HashMap<bool, (Tensor, Tensor)>,

    device: Device,

    span: tracing::Span,
}

impl FlashModernBertModel {
    pub fn load(
        vb: VarBuilder,
        config: &ModernBertConfig,
        model_type: ModelType,
        enable_fp8_dynamic: bool,
    ) -> Result<Self> {
        crate::flash_attn::validate_packed_device(&vb)?;

        let (pool, classifier) = match model_type {
            ModelType::Decision => candle::bail!("Typed decisions require a Laya checkpoint"),
            ModelType::Classifier => {
                let pool: Pool = config.classifier_pooling.clone().unwrap_or(Pool::Cls);

                let classifier: Box<dyn ClassificationHead + Send> =
                    Box::new(ModernBertClassificationHead::load(vb.clone(), config)?);

                (pool, Some(classifier))
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("`splade` is not supported for ModernBert")
                }

                (pool, None)
            }
        };

        let embeddings = ModernBertEmbeddings::load(vb.pp("model.embeddings"), config)
            .or_else(|_| ModernBertEmbeddings::load(vb.pp("embeddings"), config))?;
        let encoder = ModernBertEncoder::load(vb.pp("model.layers"), config, enable_fp8_dynamic)
            .or_else(|_| ModernBertEncoder::load(vb.pp("layers"), config, enable_fp8_dynamic))?;
        let final_norm = LayerNormNoBias::load(
            vb.pp("model.final_norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )
        .or_else(|_| {
            LayerNormNoBias::load(
                vb.pp("final_norm"),
                config.hidden_size,
                config.norm_eps as f32,
            )
        })?;

        let rotary_dim = config.hidden_size / config.num_attention_heads;
        let mut rotary_cache: HashMap<bool, (Tensor, Tensor)> = HashMap::new();

        for use_local_attention in [true, false] {
            let rope_theta = if use_local_attention {
                config.local_rope_theta
            } else {
                config.global_rope_theta
            };

            let max_position_embeddings = config.max_position_embeddings;

            let inv_freqs = get_inv_freqs(rotary_dim, rope_theta as f32, vb.device(), None)?;

            let (cos, sin) = get_cos_sin(max_position_embeddings, &inv_freqs, vb.dtype(), false)?;

            rotary_cache.insert(use_local_attention, (cos, sin));
        }

        Ok(Self {
            embeddings,
            encoder,
            final_norm,
            pool,
            classifier,
            rotary_cache,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let shape = batch.input_ids.len();
        let max_length = batch.max_length as usize;

        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;
        #[cfg(feature = "fa4")]
        let _fa4_batch =
            crate::fa4_native::prepare_batch(&cu_seqlens, &batch.cumulative_seq_lengths)?;

        let mut rotary_cache: HashMap<bool, (Tensor, Tensor)> = HashMap::new();
        for use_local_attention in [true, false] {
            let (cos, sin) = &self.rotary_cache[&use_local_attention];

            let cos = index_select(cos, &position_ids, 0)?;
            let sin = index_select(sin, &position_ids, 0)?;

            rotary_cache.insert(use_local_attention, (cos, sin));
        }

        let hidden_states = self.embeddings.forward(&input_ids)?;
        let outputs = self.encoder.forward(
            &hidden_states,
            &cu_seqlens,
            &rotary_cache,
            max_length,
            &self.final_norm,
        )?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                Pool::Cls | Pool::LastToken => {
                    if batch_size > 1 {
                        let mut indices = match self.pool {
                            Pool::Cls => cu_seqlens.narrow(0, 0, batch_size)?,
                            Pool::LastToken => {
                                let end = cu_seqlens.narrow(0, 1, batch_size)?;
                                (&end - &end.ones_like()?)?
                            }
                            _ => unreachable!(),
                        };

                        if has_raw_requests {
                            let pooled_indices = Tensor::from_vec(
                                batch.pooled_indices.clone(),
                                batch.pooled_indices.len(),
                                &self.device,
                            )?;

                            indices = index_select(&indices, &pooled_indices, 0)?
                        }

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
                let mut final_indices: Vec<u32> = Vec::with_capacity(shape);
                for i in batch.raw_indices.into_iter() {
                    let i = i as usize;
                    let start = batch.cumulative_seq_lengths[i];
                    let end = batch.cumulative_seq_lengths[i + 1];

                    for j in start..end {
                        final_indices.push(j);
                    }
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

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

impl Model for FlashModernBertModel {
    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        match &self.classifier {
            None => candle::bail!("`predict` is not implemented for this model"),
            Some(classifier) => {
                let (pooled_embeddings, _raw_embeddings) = self.forward(batch)?;
                let pooled_embeddings =
                    pooled_embeddings.expect("pooled_embeddings is empty. This is a bug.");
                classifier.forward(&pooled_embeddings)
            }
        }
    }

    fn predict_tokens(&self, batch: Batch) -> Result<Tensor> {
        match &self.classifier {
            None => candle::bail!("`predict_tokens` is not implemented for this model"),
            Some(classifier) => {
                let (_pooled_embeddings, raw_embeddings) = self.forward(batch)?;
                let raw_embeddings =
                    raw_embeddings.expect("raw_embeddings is empty. This is a bug.");
                classifier.forward_tokens(&raw_embeddings)
            }
        }
    }
}
