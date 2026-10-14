use std::collections::HashMap;

use crate::layers::{HiddenAct, LayerNormNoBias, Linear, MlpLinear};
use candle::{Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use infinity_v3_backend_core::Pool;
use serde::Deserialize;

// Preserve the configuration vocabulary. ModernBERT intentionally uses the
// tanh approximation for `gelu`, as well as its explicit approximation aliases.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModernBertActivation {
    Gelu,
    #[serde(rename = "gelu_new", alias = "gelu_pytorch_tanh")]
    GeluApprox,
    Relu,
    Silu,
    Swiglu,
    Tanh,
}

impl Module for ModernBertActivation {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => input.gelu(),
            Self::GeluApprox => input.gelu(),
            Self::Relu => input.relu(),
            Self::Silu => input.silu(),
            Self::Swiglu => candle_nn::ops::swiglu(input),
            Self::Tanh => input.tanh(),
        }
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/configuration_modernbert.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub hidden_activation: ModernBertActivation,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub initializer_cutoff_factor: f64,
    pub norm_eps: f64,
    pub norm_bias: bool,
    pub pad_token_id: usize,
    pub eos_token_id: usize,
    pub bos_token_id: usize,
    pub cls_token_id: usize,
    pub sep_token_id: usize,
    pub global_rope_theta: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub local_rope_theta: f64,
    pub embedding_dropout: Option<f64>,
    pub mlp_bias: Option<bool>,
    pub mlp_dropout: Option<f64>,
    pub decoder_bias: Option<bool>,
    pub classifier_pooling: Option<Pool>,
    pub classifier_dropout: Option<f64>,
    pub classifier_bias: Option<bool>,
    pub classifier_activation: HiddenAct,
    pub deterministic_flash_attn: Option<bool>,
    pub sparse_prediction: Option<bool>,
    pub sparse_pred_ignore_index: Option<i64>,
    pub reference_compile: Option<bool>,
    pub num_labels: Option<usize>,
    pub id2label: Option<HashMap<String, String>>,
}

#[derive(Debug)]
pub struct ModernBertEmbeddings {
    tok_embeddings: Embedding,
    norm: LayerNormNoBias,
    span: tracing::Span,
}

impl ModernBertEmbeddings {
    pub fn load(vb: VarBuilder, config: &ModernBertConfig) -> Result<Self> {
        Ok(Self {
            tok_embeddings: Embedding::new(
                vb.pp("tok_embeddings")
                    .get((config.vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ),
            norm: LayerNormNoBias::load(vb.pp("norm"), config.hidden_size, config.norm_eps as f32)?,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        self.norm
            .forward(&self.tok_embeddings.forward(input_ids)?, None)
    }
}

pub struct ModernBertMLP {
    wi: MlpLinear,
    wo: MlpLinear,
    activation: ModernBertActivation,
    span: tracing::Span,
}

impl ModernBertMLP {
    pub fn load(
        vb: VarBuilder,
        config: &ModernBertConfig,
        enable_fp8_dynamic: bool,
    ) -> Result<Self> {
        let wi_weight = vb
            .pp("Wi")
            .get((config.intermediate_size * 2, config.hidden_size), "weight")?;
        let wi_bias = vb.pp("Wi").get(config.intermediate_size * 2, "bias").ok();
        let wi = MlpLinear::with_bias_activation(wi_weight, wi_bias, None, enable_fp8_dynamic)?;

        let wo_weight = vb
            .pp("Wo")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let wo_bias = vb.pp("Wo").get(config.hidden_size, "bias").ok();

        let wo = MlpLinear::with_bias_activation(wo_weight, wo_bias, None, enable_fp8_dynamic)?;

        let activation = config.hidden_activation;

        Ok(Self {
            wi,
            wo,
            activation,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.wi.forward(hidden_states)?;

        let gated = match self.activation {
            ModernBertActivation::Gelu | ModernBertActivation::GeluApprox => {
                // Reuse the existing approximation kernel for eligible rank-two
                // Flash Attention projections. Dense rank-three inputs retain
                // the helper's activation-plus-multiply fallback.
                crate::layers::gated_activation(&hidden_states, Some(&HiddenAct::Gelu))?
            }
            _ => {
                let chunks = hidden_states.chunk(2, D::Minus1)?;
                self.activation.forward(&chunks[0])?.mul(&chunks[1])?
            }
        };
        self.wo.forward(&gated)
    }
}

pub trait ClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor>;

    // Token classification uses this hook in the CUDA model.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn forward_tokens(&self, hidden_states: &Tensor) -> Result<Tensor>;
}

pub struct ModernBertClassificationHead {
    dense: Linear,
    norm: LayerNormNoBias,
    classifier: Linear,
    span: tracing::Span,
}

impl ModernBertClassificationHead {
    pub(crate) fn load(vb: VarBuilder, config: &ModernBertConfig) -> Result<Self> {
        let n_classes = match &config.id2label {
            Some(id2label) => id2label.len(),
            None => config.num_labels.unwrap_or(1),
        };

        let dense_weight = vb
            .pp("head.dense")
            .get((config.hidden_size, config.hidden_size), "weight")?;
        let dense = Linear::new(
            dense_weight,
            None,
            Some(config.classifier_activation.clone()),
        );

        let norm = LayerNormNoBias::load(
            vb.pp("head.norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )?;

        let classifier_weight = vb
            .pp("classifier")
            .get((n_classes, config.hidden_size), "weight")?;
        let classifier_bias = vb.pp("classifier").get(n_classes, "bias")?;
        let classifier = Linear::new(classifier_weight, Some(classifier_bias), None);

        Ok(Self {
            dense,
            norm,
            classifier,
            span: tracing::span!(tracing::Level::TRACE, "classifier"),
        })
    }
}

impl ClassificationHead for ModernBertClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = hidden_states.unsqueeze(1)?;

        let hidden_states = self.dense.forward(&hidden_states)?;
        let hidden_states = self.norm.forward(&hidden_states, None)?;
        let hidden_states = self.classifier.forward(&hidden_states)?;

        let hidden_states = hidden_states.squeeze(1)?;

        Ok(hidden_states)
    }

    fn forward_tokens(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.classifier.forward(hidden_states)?;
        Ok(hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn check_approximate_gelu(device: &Device) -> Result<()> {
        // Wi yields [x, 1], so the complete gated MLP evaluates GELU(x).
        let mlp = ModernBertMLP {
            wi: MlpLinear::with_bias_activation(
                Tensor::from_slice(&[1f32, 0., 0., 1., 0., 0., 0., 0.], (4, 2), device)?,
                Some(Tensor::from_slice(&[0f32, 0., 1., 1.], 4, device)?),
                None,
                false,
            )?,
            wo: MlpLinear::with_bias_activation(
                Tensor::eye(2, DType::F32, device)?,
                None,
                None,
                false,
            )?,
            activation: serde_json::from_str("\"gelu\"").unwrap(),
            span: tracing::span!(tracing::Level::TRACE, "test_mlp"),
        };
        let output = mlp
            .forward(&Tensor::from_slice(&[-2f32, 2.], (1, 1, 2), device)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))), independently evaluated.
        // Exact erf-GELU differs by about 9.8e-5 at these inputs.
        for (actual, expected) in output.iter().zip([-0.045402307f32, 1.9545977]) {
            assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn modernbert_activation_config_preserves_existing_names() -> Result<()> {
        for (name, expected) in [
            ("gelu", ModernBertActivation::Gelu),
            ("gelu_new", ModernBertActivation::GeluApprox),
            ("gelu_pytorch_tanh", ModernBertActivation::GeluApprox),
            ("relu", ModernBertActivation::Relu),
            ("silu", ModernBertActivation::Silu),
            ("swiglu", ModernBertActivation::Swiglu),
            ("tanh", ModernBertActivation::Tanh),
        ] {
            let parsed: ModernBertActivation = serde_json::from_str(&format!("\"{name}\""))
                .expect("previously supported activation name");
            assert_eq!(parsed, expected);
        }
        Ok(())
    }

    #[test]
    fn modernbert_approximate_gelu_cpu() -> Result<()> {
        check_approximate_gelu(&Device::Cpu)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn modernbert_approximate_gelu_cuda() -> Result<()> {
        // Deliberately fail if CUDA/cuBLASLt is unavailable: this test must not
        // silently pass through the CPU fallback it is intended to distinguish.
        let device = Device::new_cuda(0)?;
        check_approximate_gelu(&device)
    }
}
