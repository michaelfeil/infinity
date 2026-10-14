use crate::layers::{HiddenAct, LayerNorm, Linear};
use candle::{Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DistilBertConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub hidden_dim: usize,
    pub activation: HiddenAct,
    pub max_position_embeddings: usize,
    pub pad_token_id: usize,
    pub model_type: Option<String>,
    pub classifier_dropout: Option<f64>,
    pub id2label: Option<HashMap<String, String>>,
}

#[derive(Debug)]
pub struct DistilBertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DistilBertEmbeddings {
    pub fn load(vb: VarBuilder, config: &DistilBertConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(
                vb.pp("word_embeddings")
                    .get((config.vocab_size, config.dim), "weight")?,
                config.dim,
            ),
            position_embeddings: Embedding::new(
                vb.pp("position_embeddings")
                    .get((config.max_position_embeddings, config.dim), "weight")?,
                config.dim,
            ),
            layer_norm: LayerNorm::load(vb.pp("LayerNorm"), config.dim, 1e-12f32)?,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(position_ids)?;

        let embeddings = self
            .layer_norm
            .forward(&input_embeddings, Some(&position_embeddings))?;

        Ok(embeddings)
    }
}

#[derive(Debug)]
pub struct DistilBertMLP {
    lin1: Linear,
    lin2: Linear,

    span: tracing::Span,
}

impl DistilBertMLP {
    pub fn load(vb: VarBuilder, config: &DistilBertConfig) -> Result<Self> {
        let lin1_weight = vb
            .pp("lin1")
            .get((config.hidden_dim, config.dim), "weight")?;
        let lin1_bias = vb.pp("lin1").get(config.hidden_dim, "bias")?;
        let lin1 = Linear::new(
            lin1_weight,
            Some(lin1_bias),
            Some(config.activation.clone()),
        );

        let lin2_weight = vb
            .pp("lin2")
            .get((config.dim, config.hidden_dim), "weight")?;
        let lin2_bias = vb.pp("lin2").get(config.dim, "bias")?;
        let lin2 = Linear::new(lin2_weight, Some(lin2_bias), None);

        Ok(Self {
            lin1,
            lin2,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.lin1.forward(hidden_states)?;
        self.lin2.forward(&hidden_states)
    }
}

pub struct DistilBertSpladeHead {
    vocab_transform: Linear,
    vocab_projector: Linear,
    vocab_layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DistilBertSpladeHead {
    pub(crate) fn load(vb: VarBuilder, config: &DistilBertConfig) -> Result<Self> {
        let vocab_transform_weight = vb
            .pp("vocab_transform")
            .get((config.dim, config.dim), "weight")?;
        let vocab_transform_bias = vb.pp("vocab_transform").get(config.dim, "bias")?;
        let vocab_transform = Linear::new(
            vocab_transform_weight,
            Some(vocab_transform_bias),
            Some(config.activation.clone()),
        );

        // When `pytorch_model.bin` originally contains `vocab_projector.weight` but the tensor
        // content shares the memory with the content on `distilbert.embeddings.word_embeddings.weight`,
        // e.g. a subset of the original tensor, when converting the file from BIN to Safentensors
        // the latter tensor that shares the memory with the previous will be removed
        let vocab_projector_weight = if vb.contains_tensor("vocab_projector.weight") {
            vb.pp("vocab_projector")
                .get((config.vocab_size, config.dim), "weight")?
        } else {
            vb.pp("distilbert.embeddings.word_embeddings")
                .get((config.vocab_size, config.dim), "weight")?
        };
        let vocab_projector_bias = vb.pp("vocab_projector").get(config.vocab_size, "bias")?;
        let vocab_projector = Linear::new(
            vocab_projector_weight,
            Some(vocab_projector_bias),
            Some(HiddenAct::Relu),
        );

        let vocab_layer_norm = LayerNorm::load(vb.pp("vocab_layer_norm"), config.dim, 1e-12f32)?;

        Ok(Self {
            vocab_transform,
            vocab_projector,
            vocab_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "splade"),
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.vocab_transform.forward(hidden_states)?;
        let hidden_states = self.vocab_layer_norm.forward(&hidden_states, None)?;
        let hidden_states = self.vocab_projector.forward(&hidden_states)?;
        (1.0 + hidden_states)?.log()
    }
}
