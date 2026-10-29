use crate::layers::HiddenAct;
use candle::{Device, Result, Tensor};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma3RopeScaling {
    #[serde(alias = "type")]
    pub rope_type: String,
    #[serde(default = "default_rope_factor")]
    pub factor: f32,
}

fn default_rope_factor() -> f32 {
    1.0
}

impl Gemma3RopeScaling {
    #[cfg_attr(not(feature = "flash-attn"), allow(dead_code))]
    fn factor(&self) -> Result<Option<f32>> {
        match self.rope_type.as_str() {
            "default" => Ok(None),
            "linear" => {
                if !self.factor.is_finite() || self.factor <= 0.0 {
                    candle::bail!("Gemma3 linear RoPE scaling factor must be positive");
                }
                Ok(Some(self.factor))
            }
            other => candle::bail!("Unsupported Gemma3 RoPE scaling type `{other}`"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma3Config {
    pub attention_bias: bool,
    #[serde(default)]
    pub use_bidirectional_attention: bool,
    pub pad_token_id: u32,
    pub head_dim: Option<usize>,
    pub hidden_activation: HiddenAct,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub query_pre_attn_scalar: usize,
    pub rms_norm_eps: f32,
    pub rope_local_base_freq: f32,
    #[serde(default)]
    pub rope_scaling: Option<Gemma3RopeScaling>,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
    #[serde(rename(deserialize = "_sliding_window_pattern"))]
    pub sliding_window_pattern: usize,
    pub vocab_size: usize,
}

#[cfg_attr(not(feature = "flash-attn"), allow(dead_code))]
fn global_inv_freqs(config: &Gemma3Config, dim: usize, device: &Device) -> Result<Tensor> {
    let inv_freqs = crate::layers::get_inv_freqs(dim, config.rope_theta, device, None)?;
    match config
        .rope_scaling
        .as_ref()
        .map(Gemma3RopeScaling::factor)
        .transpose()?
        .flatten()
    {
        Some(factor) => Ok((inv_freqs / factor as f64)?),
        None => Ok(inv_freqs),
    }
}

#[cfg(feature = "flash-attn")]
mod packed {
    use super::*;
    use crate::layers::{get_cos_sin, get_inv_freqs, Linear};
    use crate::models::Model;
    use candle::DType;
    use candle_nn::{Embedding, Module, VarBuilder};
    use infinity_v3_backend_core::{Batch, ModelType, Pool};
    #[derive(Debug)]
    pub struct Gemma3RMSNorm {
        scale: Tensor,
        epsilon: f32,

        span: tracing::Span,
    }

    impl Gemma3RMSNorm {
        pub fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
            Ok(Self {
                scale: (vb
                    .get(hidden_size, "weight")
                    .or_else(|_| vb.get(hidden_size, "gamma"))?
                    .to_dtype(DType::F32)?
                    + 1.0)?,
                epsilon,
                span: tracing::span!(tracing::Level::TRACE, "rms-norm"),
            })
        }

        pub fn forward(
            &self,
            hidden_states: &Tensor,
            residual: Option<&Tensor>,
        ) -> Result<(Tensor, Tensor)> {
            let _enter = self.span.enter();

            // Gemma multiplies the FP32 normalized value by (1 + weight) before
            // rounding to the activation dtype. Pre-rounding the scale loses bits.
            let residual_add = match residual {
                Some(residual) => hidden_states.add(residual)?,
                None => hidden_states.clone(),
            };
            let normalized =
                crate::layers::gemma_rms_norm::forward(&residual_add, &self.scale, self.epsilon)?;
            Ok((normalized, residual_add))
        }
    }

    enum Gemma3AttentionType {
        FullAttention,
        SlidingAttention,
    }

    struct Gemma3Attention {
        qkv_proj: Linear,
        o_proj: Linear,

        q_norm: Gemma3RMSNorm,
        k_norm: Gemma3RMSNorm,

        attention_head_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        scaling: f64,

        sliding_window: Option<usize>,
        causal: bool,

        span: tracing::Span,
    }

    impl Gemma3Attention {
        pub fn load(
            vb: VarBuilder,
            config: &Gemma3Config,
            attention_type: Gemma3AttentionType,
        ) -> Result<Self> {
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
            let key_weight = vb.pp("k_proj").get(
                (num_key_value_heads * attention_head_size, hidden_size),
                "weight",
            )?;
            let value_weight = vb.pp("v_proj").get(
                (num_key_value_heads * attention_head_size, hidden_size),
                "weight",
            )?;

            let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;

            let qkv_bias = if config.attention_bias {
                let query_bias = vb
                    .pp("q_proj")
                    .get(num_attention_heads * attention_head_size, "bias")?;
                let key_bias = vb
                    .pp("k_proj")
                    .get(num_key_value_heads * attention_head_size, "bias")?;
                let value_bias = vb
                    .pp("v_proj")
                    .get(num_key_value_heads * attention_head_size, "bias")?;
                Some(Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?)
            } else {
                None
            };

            let qkv_proj = Linear::new(qkv_weight, qkv_bias, None);

            let output_weight = vb.pp("o_proj").get(
                (hidden_size, num_attention_heads * attention_head_size),
                "weight",
            )?;
            let output_bias = if config.attention_bias {
                Some(vb.pp("o_proj").get(hidden_size, "bias")?)
            } else {
                None
            };
            let o_proj = Linear::new(output_weight, output_bias, None);

            let q_norm =
                Gemma3RMSNorm::load(vb.pp("q_norm"), attention_head_size, config.rms_norm_eps)?;
            let k_norm =
                Gemma3RMSNorm::load(vb.pp("k_norm"), attention_head_size, config.rms_norm_eps)?;

            let scaling = 1.0 / (config.query_pre_attn_scalar as f64).sqrt();

            match attention_type {
                Gemma3AttentionType::FullAttention => Ok(Self {
                    qkv_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    attention_head_size,
                    num_attention_heads,
                    num_key_value_heads,
                    scaling,
                    sliding_window: None,
                    causal: !config.use_bidirectional_attention,
                    span: tracing::span!(tracing::Level::TRACE, "full_attention"),
                }),
                Gemma3AttentionType::SlidingAttention => Ok(Self {
                    qkv_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    attention_head_size,
                    num_attention_heads,
                    num_key_value_heads,
                    scaling,
                    sliding_window: config.sliding_window,
                    causal: !config.use_bidirectional_attention,
                    span: tracing::span!(tracing::Level::TRACE, "sliding_attention"),
                }),
            }
        }
    }

    #[cfg(feature = "flash-attn")]
    impl Gemma3Attention {
        fn forward_packed(
            &self,
            states: &Tensor,
            cumulative: &Tensor,
            max_length: usize,
            cos: &Tensor,
            sin: &Tensor,
        ) -> Result<Tensor> {
            let _enter = self.span.enter();
            let qkv = self.qkv_proj.forward(states)?;
            let tokens = states.dim(0)?;
            let q_size = self.num_attention_heads * self.attention_head_size;
            // Reshape the contiguous projection before slicing heads: Candle's
            // reshape of a strided slice would materialize a copy.
            let qkv = qkv.reshape((
                tokens,
                self.num_attention_heads + 2 * self.num_key_value_heads,
                self.attention_head_size,
            ))?;
            let q = qkv.narrow(1, 0, self.num_attention_heads)?;
            let k = qkv.narrow(1, self.num_attention_heads, self.num_key_value_heads)?;
            let v = qkv.narrow(
                1,
                self.num_attention_heads + self.num_key_value_heads,
                self.num_key_value_heads,
            )?;
            let (q, _) = self.q_norm.forward(&q, None)?;
            let (k, _) = self.k_norm.forward(&k, None)?;
            // RMSNorm produces fresh contiguous Q/K buffers. The existing CUDA
            // kernel rotates both in place using half-width, packed NeoX caches.
            candle_rotary::apply_rotary_inplace(&q, &k, cos, sin, true)?;
            let radius = self
                .sliding_window
                .map(|w| if self.causal { w - 1 } else { w / 2 });
            let output = crate::flash_attn::flash_attn_varlen(
                &q,
                &k,
                &v,
                None,
                cumulative,
                cumulative,
                max_length,
                max_length,
                self.scaling as f32,
                self.causal,
                radius,
                radius.map(|r| if self.causal { 0 } else { r }),
            )?;
            self.o_proj.forward(&output.reshape((tokens, q_size))?)
        }
    }

    struct Gemma3MLP {
        gate_up_proj: Linear,
        down_proj: Linear,
        hidden_activation: HiddenAct,

        span: tracing::Span,
    }

    impl Gemma3MLP {
        pub fn load(vb: VarBuilder, config: &Gemma3Config) -> Result<Self> {
            let gate_proj_weight = vb
                .pp("gate_proj")
                .get((config.intermediate_size, config.hidden_size), "weight")?;

            let up_proj_weight = vb
                .pp("up_proj")
                .get((config.intermediate_size, config.hidden_size), "weight")?;

            let gate_up_proj_weight = Tensor::cat(&[&gate_proj_weight, &up_proj_weight], 0)?;
            let gate_up_proj = Linear::new(gate_up_proj_weight, None, None);

            let down_proj_weight = vb
                .pp("down_proj")
                .get((config.hidden_size, config.intermediate_size), "weight")?;
            let down_proj = Linear::new(down_proj_weight, None, None);

            Ok(Self {
                gate_up_proj,
                down_proj,
                hidden_activation: config.hidden_activation.clone(),
                span: tracing::span!(tracing::Level::TRACE, "mlp"),
            })
        }

        pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
            let _enter = self.span.enter();

            let gate_up_states = self.gate_up_proj.forward(hidden_states)?;

            let activated =
                crate::layers::gated_activation(&gate_up_states, Some(&self.hidden_activation))?;
            self.down_proj.forward(&activated)
        }
    }

    struct Gemma3Layer {
        input_layernorm: Gemma3RMSNorm,
        self_attn: Gemma3Attention,
        post_attention_layernorm: Gemma3RMSNorm,

        pre_feedforward_layernorm: Gemma3RMSNorm,
        mlp: Gemma3MLP,
        post_feedforward_layernorm: Gemma3RMSNorm,

        span: tracing::Span,
    }

    impl Gemma3Layer {
        pub fn load(
            vb: VarBuilder,
            config: &Gemma3Config,
            attention_type: Gemma3AttentionType,
        ) -> Result<Self> {
            let input_layernorm = Gemma3RMSNorm::load(
                vb.pp("input_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?;
            let self_attn = Gemma3Attention::load(vb.pp("self_attn"), config, attention_type)?;
            let post_attention_layernorm = Gemma3RMSNorm::load(
                vb.pp("post_attention_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?;

            let pre_feedforward_layernorm = Gemma3RMSNorm::load(
                vb.pp("pre_feedforward_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?;
            let mlp = Gemma3MLP::load(vb.pp("mlp"), config)?;
            let post_feedforward_layernorm = Gemma3RMSNorm::load(
                vb.pp("post_feedforward_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?;

            Ok(Self {
                input_layernorm,
                self_attn,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                mlp,
                post_feedforward_layernorm,
                span: tracing::span!(tracing::Level::TRACE, "layer"),
            })
        }
    }

    #[cfg(feature = "flash-attn")]
    impl Gemma3Layer {
        fn forward_packed(
            &self,
            states: &Tensor,
            cumulative: &Tensor,
            max_length: usize,
            cos: &Tensor,
            sin: &Tensor,
        ) -> Result<Tensor> {
            let _enter = self.span.enter();
            let (normalized, _) = self.input_layernorm.forward(states, None)?;
            let attention =
                self.self_attn
                    .forward_packed(&normalized, cumulative, max_length, cos, sin)?;
            let (attention, _) = self.post_attention_layernorm.forward(&attention, None)?;
            let states = crate::layers::residual_add(states, &attention)?;
            let (normalized, _) = self.pre_feedforward_layernorm.forward(&states, None)?;
            let mlp = self.mlp.forward(&normalized)?;
            let (mlp, _) = self.post_feedforward_layernorm.forward(&mlp, None)?;
            crate::layers::residual_add(&states, &mlp)
        }
    }

    pub struct Gemma3Embedding {
        embedding: Embedding,
        scale: Tensor,

        span: tracing::Span,
    }

    impl Gemma3Embedding {
        pub fn load(vb: VarBuilder, config: &Gemma3Config) -> Result<Self> {
            let embedding = Embedding::new(
                vb.get((config.vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            );
            let scale = Tensor::new((config.hidden_size as f32).sqrt(), vb.device())?
                .to_dtype(vb.dtype())?;

            Ok(Self {
                embedding,
                scale,
                span: tracing::span!(tracing::Level::TRACE, "embed_tokens"),
            })
        }

        pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
            let _enter = self.span.enter();

            let hidden_states = self.embedding.forward(input_ids)?;
            hidden_states.broadcast_mul(&self.scale)
        }
    }

    pub struct Gemma3Model {
        embed_tokens: Gemma3Embedding,
        layers: Vec<Gemma3Layer>,
        norm: Gemma3RMSNorm,

        rotary_cache: (Tensor, Tensor),
        rotary_cache_local_attention: (Tensor, Tensor),
        rotary_dim: usize,
        pool: Pool,

        device: Device,

        span: tracing::Span,
    }

    impl Gemma3Model {
        pub fn load(vb: VarBuilder, config: &Gemma3Config, model_type: ModelType) -> Result<Self> {
            if config.sliding_window_pattern == 0
                || config.sliding_window == Some(0)
                || config.num_key_value_heads == 0
                || config.num_attention_heads == 0
                || !config
                    .num_attention_heads
                    .is_multiple_of(config.num_key_value_heads)
                || config.query_pre_attn_scalar == 0
            {
                candle::bail!("Invalid Gemma3 attention configuration");
            }
            if vb.dtype() != DType::BF16 || !vb.device().is_cuda() {
                candle::bail!("EmbeddingGemma requires CUDA bfloat16 with packed FlashAttention; float32 and float16 model execution are unsupported");
            }
            if std::env::var("USE_FLASH_ATTENTION")
                .unwrap_or_else(|_| "true".into())
                .to_lowercase()
                != "true"
            {
                candle::bail!("EmbeddingGemma requires packed FlashAttention; USE_FLASH_ATTENTION=false is unsupported");
            }
            let vb = if vb.contains_tensor("model.embed_tokens.weight") {
                vb.pp("model")
            } else {
                vb
            };
            let pool = match model_type {
                ModelType::Decision => candle::bail!("Typed decisions require a Laya checkpoint"),
                ModelType::Classifier => {
                    candle::bail!("`classifier` model type is not supported for Gemma3")
                }
                ModelType::Embedding(Pool::Splade) => {
                    candle::bail!("SPLADE pooling is not supported for Gemma3")
                }
                ModelType::Embedding(pool) => pool,
            };

            let embed_tokens = Gemma3Embedding::load(vb.pp("embed_tokens"), config)?;

            let layers = (0..config.num_hidden_layers)
                .map(|layer_idx| {
                    let attention_type = match (layer_idx + 1) % config.sliding_window_pattern > 0 {
                        false => Gemma3AttentionType::FullAttention,
                        true => Gemma3AttentionType::SlidingAttention,
                    };
                    Gemma3Layer::load(vb.pp(format!("layers.{layer_idx}")), config, attention_type)
                })
                .collect::<Result<Vec<Gemma3Layer>>>()?;

            let norm = Gemma3RMSNorm::load(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

            let rotary_dim = config
                .head_dim
                .unwrap_or(config.hidden_size / config.num_attention_heads);

            let inv_freqs = global_inv_freqs(config, rotary_dim, vb.device())?;
            let rotary_cache = get_cos_sin(
                config.max_position_embeddings,
                &inv_freqs,
                vb.dtype(),
                false,
            )?;

            let inv_freqs_local =
                get_inv_freqs(rotary_dim, config.rope_local_base_freq, vb.device(), None)?;
            let rotary_cache_local_attention = get_cos_sin(
                config.max_position_embeddings,
                &inv_freqs_local,
                vb.dtype(),
                false,
            )?;

            Ok(Self {
                embed_tokens,
                layers,
                norm,
                rotary_cache,
                rotary_cache_local_attention,
                rotary_dim,
                pool,
                device: vb.device().clone(),
                span: tracing::span!(tracing::Level::TRACE, "model"),
            })
        }
    }

    #[cfg(feature = "flash-attn")]
    impl Gemma3Model {
        fn forward_packed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
            let tokens = batch.input_ids.len();
            let ids = Tensor::from_vec(batch.input_ids, tokens, &self.device)?;
            let positions = Tensor::from_vec(batch.position_ids, tokens, &self.device)?;
            let cumulative = Tensor::from_vec(
                batch.cumulative_seq_lengths.clone(),
                batch.cumulative_seq_lengths.len(),
                &self.device,
            )?;
            #[cfg(feature = "fa4")]
            let _fa4_batch =
                crate::fa4_native::prepare_batch(&cumulative, &batch.cumulative_seq_lengths)?;
            let gather = |cache: &(Tensor, Tensor)| -> Result<(Tensor, Tensor)> {
                Ok((
                    cache
                        .0
                        .index_select(&positions, 0)?
                        .reshape((tokens, self.rotary_dim / 2))?,
                    cache
                        .1
                        .index_select(&positions, 0)?
                        .reshape((tokens, self.rotary_dim / 2))?,
                ))
            };
            let (cos, sin) = gather(&self.rotary_cache)?;
            let (local_cos, local_sin) = gather(&self.rotary_cache_local_attention)?;
            let mut states = self.embed_tokens.forward(&ids)?;
            for layer in &self.layers {
                let (cos, sin) = if layer.self_attn.sliding_window.is_some() {
                    (&local_cos, &local_sin)
                } else {
                    (&cos, &sin)
                };
                states = layer.forward_packed(
                    &states,
                    &cumulative,
                    batch.max_length as usize,
                    cos,
                    sin,
                )?;
            }
            let (outputs, _) = self.norm.forward(&states, None)?;
            let pooled = if batch.pooled_indices.is_empty() {
                None
            } else if self.pool == Pool::Mean {
                Some(crate::layers::mean_pool(
                    &outputs,
                    &batch.cumulative_seq_lengths,
                    &batch.pooled_indices,
                )?)
            } else {
                let rows = batch
                    .pooled_indices
                    .iter()
                    .map(|&index| -> Result<Tensor> {
                        let index = index as usize;
                        let (Some(&start), Some(&end)) = (
                            batch.cumulative_seq_lengths.get(index),
                            batch.cumulative_seq_lengths.get(index + 1),
                        ) else {
                            candle::bail!("Gemma3 pooling sequence index out of range");
                        };
                        let (start, end) = (start as usize, end as usize);
                        if end <= start {
                            candle::bail!("Gemma3 pooling requires nonempty, in-bounds sequences");
                        }
                        match self.pool {
                            Pool::Cls => outputs.narrow(0, start, 1),
                            Pool::LastToken => outputs.narrow(0, end - 1, 1),
                            Pool::Mean | Pool::Splade => {
                                candle::bail!("Unsupported Gemma3 row pooling mode")
                            }
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                Some(Tensor::cat(&rows, 0)?)
            };
            let raw = if batch.raw_indices.is_empty() {
                None
            } else {
                let rows: Result<Vec<_>> = batch
                    .raw_indices
                    .iter()
                    .map(|&i| {
                        let start = batch.cumulative_seq_lengths[i as usize] as usize;
                        let end = batch.cumulative_seq_lengths[i as usize + 1] as usize;
                        outputs.narrow(0, start, end - start)
                    })
                    .collect();
                Some(Tensor::cat(&rows?, 0)?)
            };
            Ok((pooled, raw))
        }
    }

    impl Model for Gemma3Model {
        fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
            let _enter = self.span.enter();
            self.forward_packed(batch)
        }
    }
}

#[cfg(feature = "flash-attn")]
pub use packed::Gemma3Model;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config(rope_scaling: serde_json::Value) -> Gemma3Config {
        serde_json::from_value(json!({
            "attention_bias": false,
            "pad_token_id": 0,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 3840,
            "intermediate_size": 15360,
            "max_position_embeddings": 131072,
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "num_key_value_heads": 8,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-6,
            "rope_local_base_freq": 10000.0,
            "rope_scaling": rope_scaling,
            "rope_theta": 1000000.0,
            "sliding_window": 1024,
            "_sliding_window_pattern": 6,
            "vocab_size": 262208
        }))
        .unwrap()
    }

    #[test]
    fn kalm_linear_rope_scaling_divides_inverse_frequencies() {
        let config = config(json!({"factor": 8.0, "rope_type": "linear"}));
        let frequencies = global_inv_freqs(&config, 256, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (index, actual) in frequencies.into_iter().enumerate() {
            let expected = (1.0 / 1_000_000f32.powf(2.0 * index as f32 / 256.0)) / 8.0;
            assert!((actual - expected).abs() <= expected.abs() * 1e-6);
        }
    }

    #[test]
    fn rejects_unknown_gemma3_rope_scaling() {
        let config = config(json!({"factor": 8.0, "rope_type": "yarn"}));
        assert!(global_inv_freqs(&config, 256, &Device::Cpu).is_err());
    }
}
