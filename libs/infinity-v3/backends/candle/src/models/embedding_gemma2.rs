//! Bidirectional EmbeddingGemma 2 encoder with shared Gemma4 modality towers.
use super::gemma4::{Gemma4Layer, Gemma4RmsNorm, Gemma4TextConfig, SharedKv};
use super::Model;
use crate::layers::{get_cos_sin, get_inv_freqs, CompactUnfoldTensors, Linear};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use infinity_v3_backend_core::{Batch, ModelType, Pool};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct EmbeddingGemma2Config {
    pub text_config: EmbeddingGemma2TextConfig,
    pub vision_config: Option<serde_json::Value>,
    pub audio_config: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingGemma2TextConfig {
    #[serde(flatten)]
    backbone: Gemma4TextConfig,
    embedding_dim: usize,
    #[serde(default)]
    per_layer_config: HashMap<String, AttentionConfig>,
}

#[derive(Debug, Deserialize)]
struct AttentionConfig {
    head_dim: Option<usize>,
    num_key_value_heads: Option<usize>,
}

impl EmbeddingGemma2TextConfig {
    fn layer_config(&self, idx: usize) -> Result<Gemma4TextConfig> {
        let mut config = self.backbone.clone();
        // HF serializes layer indices with leading zeroes (e.g. "05").
        let overrides = self
            .per_layer_config
            .iter()
            .find_map(|(key, value)| (key.parse::<usize>().ok() == Some(idx)).then_some(value));
        let head_dim = overrides
            .and_then(|c| c.head_dim)
            .unwrap_or(config.head_dim);
        config.head_dim = head_dim;
        config.global_head_dim = head_dim;
        config.num_key_value_heads = overrides
            .and_then(|c| c.num_key_value_heads)
            .unwrap_or(config.num_key_value_heads);
        // Gemma4 uses half the window for bidirectional attention; this encoder
        // defines sliding_window as the inclusive radius on each side.
        config.sliding_window = config
            .sliding_window
            .checked_mul(2)
            .ok_or_else(|| candle::Error::Msg("EmbeddingGemma2 sliding window overflows".into()))?;
        Ok(config)
    }
}

pub struct EmbeddingGemma2Model {
    #[cfg(feature = "flash-attn")]
    vision: Option<super::gemma4_vision::Gemma4Vision>,
    #[cfg(feature = "flash-attn")]
    audio: Option<candle_transformers::models::gemma4::AudioModel>,
    #[cfg(feature = "flash-attn")]
    audio_projection:
        Option<candle_transformers::models::gemma4::multimodal_embedding::MultimodalEmbedder>,
    embeddings: Embedding,
    embedding_scale: Tensor,
    ple_projection: Linear,
    ple_norm: Gemma4RmsNorm,
    ple_scale: f64,
    ple_width: usize,
    layers: Vec<Gemma4Layer>,
    norm: Gemma4RmsNorm,
    projection: Linear,
    inv_freqs: Vec<Tensor>,
    rope_layers: Vec<usize>,
    max_positions: usize,
    pool: Pool,
    device: Device,
}

impl EmbeddingGemma2Model {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2Config,
        model_type: ModelType,
    ) -> Result<Self> {
        if !vb.device().is_cuda() || vb.dtype() != DType::BF16 || !cfg!(feature = "flash-attn") {
            candle::bail!("EmbeddingGemma2 requires CUDA BF16 with FlashAttention v2");
        }
        let ModelType::Embedding(pool) = model_type else {
            candle::bail!("EmbeddingGemma2 only supports embeddings");
        };
        if pool == Pool::Splade {
            candle::bail!("Splade pooling is not supported for EmbeddingGemma2");
        }
        let text = &config.text_config.backbone;
        if text.num_hidden_layers == 0
            || config.text_config.embedding_dim == 0
            || text.layer_types.len() != text.num_hidden_layers
            || text.hidden_size_per_layer_input == 0
        {
            candle::bail!("Invalid EmbeddingGemma2 layer count or per-layer input width");
        }
        #[cfg(feature = "flash-attn")]
        let modality_vb = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            vb.clone()
        };
        #[cfg(feature = "flash-attn")]
        let vision = config
            .vision_config
            .as_ref()
            .map(|c| {
                super::gemma4_vision::Gemma4Vision::load_at(
                    modality_vb.clone(),
                    c,
                    text.hidden_size,
                    "",
                )
            })
            .transpose()?;
        #[cfg(feature = "flash-attn")]
        let (audio, audio_projection) = match &config.audio_config {
            Some(c) => {
                let mut cfg: candle_transformers::models::gemma4::config::Gemma4AudioConfig =
                    serde_json::from_value(c.clone()).map_err(candle::Error::wrap)?;
                cfg.conf_reduction_factor = 1;
                cfg.gradient_clipping = cfg.gradient_clipping.min(f32::MAX as f64);
                let projection = candle_transformers::models::gemma4::multimodal_embedding::MultimodalEmbedder::new(
                    cfg.output_proj_dims.unwrap_or(cfg.hidden_size), text.hidden_size, cfg.rms_norm_eps,
                    modality_vb.pp("embed_audio"),
                )?;
                (
                    Some(candle_transformers::models::gemma4::AudioModel::new(
                        &cfg,
                        modality_vb.pp("audio_tower"),
                    )?),
                    Some(projection),
                )
            }
            None => (None, None),
        };
        let vb = if vb.contains_tensor("language_model.embed_tokens.weight") {
            vb.pp("language_model")
        } else if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model.language_model")
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            vb
        };
        let linear = |name: &str, output: usize, input: usize| -> Result<Linear> {
            Ok(Linear::new(
                vb.pp(name).get((output, input), "weight")?,
                None,
                None,
            ))
        };
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        let mut inv_freqs = Vec::new();
        let mut rope_keys = Vec::new();
        let mut rope_layers = Vec::with_capacity(text.num_hidden_layers);
        for idx in 0..text.num_hidden_layers {
            let layer_config = config.text_config.layer_config(idx)?;
            if layer_config.head_dim == 0
                || layer_config.head_dim > 512
                || layer_config.head_dim % 8 != 0
                || layer_config.num_key_value_heads == 0
                || layer_config.num_attention_heads == 0
                || layer_config.num_attention_heads % layer_config.num_key_value_heads != 0
            {
                candle::bail!("Unsupported EmbeddingGemma2 attention geometry at layer {idx}");
            }
            let layer_vb = vb.pp(format!("layers.{idx}"));
            layers.push(Gemma4Layer::load_with_ple(
                layer_vb.clone(),
                layer_vb.pp("ple_block"),
                &layer_config,
                idx,
            )?);
            let theta = match text.layer_types[idx].as_str() {
                "sliding_attention" => text.rope_parameters.sliding_attention.rope_theta,
                "full_attention" => text.rope_parameters.full_attention.rope_theta,
                other => candle::bail!("Unsupported EmbeddingGemma2 attention type {other}"),
            };
            let key = (layer_config.head_dim, theta.to_bits());
            let rope_idx = match rope_keys.iter().position(|existing| *existing == key) {
                Some(idx) => idx,
                None => {
                    rope_keys.push(key);
                    inv_freqs.push(get_inv_freqs(
                        layer_config.head_dim,
                        theta,
                        vb.device(),
                        None,
                    )?);
                    inv_freqs.len() - 1
                }
            };
            rope_layers.push(rope_idx);
        }
        Ok(Self {
            #[cfg(feature = "flash-attn")]
            vision,
            #[cfg(feature = "flash-attn")]
            audio,
            #[cfg(feature = "flash-attn")]
            audio_projection,
            embeddings: Embedding::new(
                vb.pp("embed_tokens")
                    .get((text.vocab_size, text.hidden_size), "weight")?,
                text.hidden_size,
            ),
            // The reference casts sqrt(hidden_size) to BF16 before multiplying.
            embedding_scale: Tensor::new((text.hidden_size as f32).sqrt(), vb.device())?
                .to_dtype(vb.dtype())?,
            ple_projection: linear(
                "ple.per_layer_model_projection",
                text.num_hidden_layers * text.hidden_size_per_layer_input,
                text.hidden_size,
            )?,
            ple_norm: Gemma4RmsNorm::load(
                vb.pp("ple.per_layer_projection_norm"),
                text.hidden_size_per_layer_input,
                text.rms_norm_eps,
            )?,
            ple_scale: (text.hidden_size as f64).sqrt().recip(),
            ple_width: text.hidden_size_per_layer_input,
            layers,
            norm: Gemma4RmsNorm::load(vb.pp("norm"), text.hidden_size, text.rms_norm_eps)?,
            projection: linear(
                "embedding_projection",
                config.text_config.embedding_dim,
                text.hidden_size,
            )?,
            inv_freqs,
            rope_layers,
            max_positions: text.max_position_embeddings,
            pool,
            device: vb.device().clone(),
        })
    }

    #[cfg(feature = "flash-attn")]
    fn forward(&self, batch: &Batch) -> Result<Tensor> {
        if batch.compact_input_ids.is_some() {
            candle::bail!("Bidirectional EmbeddingGemma2 cannot fold causal prefixes");
        }
        let (ids, compact) = CompactUnfoldTensors::from_batch(batch, &self.device)?;
        let mut states = self
            .embeddings
            .forward(&ids)?
            .broadcast_mul(&self.embedding_scale)?;
        if batch
            .multimodal
            .iter()
            .flatten()
            .any(|media| !media.images.is_empty())
        {
            self.vision
                .as_ref()
                .ok_or_else(|| {
                    candle::Error::Msg("EmbeddingGemma2 vision tower unavailable".into())
                })?
                .inject(batch, &mut states)?;
        }
        for (row, media) in batch.multimodal.iter().enumerate() {
            let Some(media) = media else { continue };
            let base = batch.cumulative_seq_lengths[row] as usize;
            let seq_len = (batch.cumulative_seq_lengths[row + 1]
                - batch.cumulative_seq_lengths[row]) as usize;
            for (start, features) in &media.audios {
                if features.feature_size != 128
                    || features.values.len() != features.mask.len() * 128
                    || start + features.token_count() > seq_len
                {
                    candle::bail!("Invalid EmbeddingGemma2 audio features");
                }
                let input = Tensor::from_slice(
                    &features.values,
                    (1, features.mask.len(), 128),
                    &self.device,
                )?
                .to_dtype(states.dtype())?;
                // The shared conformer uses 0 for valid and 1 for padding.
                let mask: Vec<f32> = features
                    .mask
                    .iter()
                    .map(|&v| if v != 0 { 0.0 } else { 1.0 })
                    .collect();
                let mask = Tensor::new(mask.as_slice(), &self.device)?.unsqueeze(0)?;
                let tower = self.audio.as_ref().ok_or_else(|| {
                    candle::Error::Msg("EmbeddingGemma2 audio tower unavailable".into())
                })?;
                let (encoded, _) = tower.forward(&input, &mask)?;
                let projected = self
                    .audio_projection
                    .as_ref()
                    .unwrap()
                    .forward(&encoded)?
                    .squeeze(0)?;
                let indices: Vec<u32> = features
                    .mask
                    .iter()
                    .step_by(4)
                    .enumerate()
                    .filter_map(|(i, &v)| (v != 0).then_some(i as u32))
                    .collect();
                let indices = Tensor::new(indices.as_slice(), &self.device)?;
                let projected = crate::layers::index_select(&projected, &indices, 0)?;
                states = states.slice_scatter0(&projected, base + start)?;
            }
        }
        let ple = (self.ple_projection.forward(&states)? * self.ple_scale)?.reshape((
            ids.dim(0)?,
            self.layers.len(),
            self.ple_width,
        ))?;
        let ple = self.ple_norm.forward(&ple)?;
        let cu = Tensor::new(batch.cumulative_seq_lengths.as_slice(), &self.device)?;
        #[cfg(feature = "fa4")]
        let _fa4_batch = crate::fa4_native::prepare_batch(&cu, &batch.cumulative_seq_lengths)?;
        let rope_len = batch.position_ids.iter().copied().max().unwrap_or(0) as usize + 1;
        if rope_len > self.max_positions {
            candle::bail!("EmbeddingGemma2 position exceeds max_position_embeddings");
        }
        // Cache positions once per attention geometry for this batch. The
        // released model has only two geometries despite its 24 layers.
        let select = |x: Tensor| -> Result<Tensor> {
            crate::layers::index_select(&x, &compact.position_ids_compact, 0)?
                .unsqueeze(0)?
                .unsqueeze(0)
        };
        let rope = self
            .inv_freqs
            .iter()
            .map(|freqs| {
                let (cos, sin) = get_cos_sin(rope_len, freqs, states.dtype(), true)?;
                Ok((select(cos)?, select(sin)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut shared_kv = SharedKv::default();
        for (idx, layer) in self.layers.iter().enumerate() {
            let (cos, sin) = &rope[self.rope_layers[idx]];
            states = layer.forward_varlen(
                &states,
                Some(&ple.i((.., idx, ..))?),
                cos,
                sin,
                &cu,
                batch.max_length as usize,
                false,
                &compact,
                &mut shared_kv,
                &[],
                None,
            )?;
        }
        self.projection.forward(&self.norm.forward(&states)?)
    }
}

impl Model for EmbeddingGemma2Model {
    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        #[cfg(feature = "flash-attn")]
        {
            let states = self.forward(&batch)?;
            let pooled = if batch.pooled_indices.is_empty() {
                None
            } else {
                let values = batch
                    .pooled_indices
                    .iter()
                    .map(|&idx| {
                        let start = batch.cumulative_seq_lengths[idx as usize] as usize;
                        let end = batch.cumulative_seq_lengths[idx as usize + 1] as usize;
                        match self.pool {
                            Pool::Cls => states.i(start)?.unsqueeze(0),
                            Pool::LastToken => states.i(end - 1)?.unsqueeze(0),
                            Pool::Mean => {
                                states.narrow(0, start, end - start)?.sum_keepdim(0)?
                                    / (end - start) as f64
                            }
                            Pool::Splade => {
                                candle::bail!("Splade pooling is not supported for EmbeddingGemma2")
                            }
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                Some(Tensor::cat(&values, 0)?)
            };
            let raw = if batch.raw_indices.is_empty() {
                None
            } else {
                let values = batch
                    .raw_indices
                    .iter()
                    .map(|&idx| {
                        let start = batch.cumulative_seq_lengths[idx as usize] as usize;
                        let end = batch.cumulative_seq_lengths[idx as usize + 1] as usize;
                        states.narrow(0, start, end - start)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Some(Tensor::cat(&values, 0)?)
            };
            Ok((pooled, raw))
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            let _ = batch;
            candle::bail!("EmbeddingGemma2 requires CUDA BF16 with FlashAttention v2");
        }
    }

    fn predict(&self, _batch: Batch) -> Result<Tensor> {
        candle::bail!("EmbeddingGemma2 only supports embeddings");
    }
}
