use crate::layers::{LayerNorm, Linear};
use crate::models::BertConfig;
use crate::models::PositionEmbeddingType;
use candle::{Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};

#[derive(Debug)]
pub struct JinaEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl JinaEmbeddings {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let position_embeddings =
            if config.position_embedding_type == PositionEmbeddingType::Absolute {
                Some(Embedding::new(
                    vb.pp("position_embeddings").get(
                        (config.max_position_embeddings, config.hidden_size),
                        "weight",
                    )?,
                    config.hidden_size,
                ))
            } else {
                None
            };

        Ok(Self {
            word_embeddings: Embedding::new(
                vb.pp("word_embeddings")
                    .get((config.vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ),
            token_type_embeddings: Embedding::new(
                vb.pp("token_type_embeddings")
                    .get((config.type_vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ),
            position_embeddings,
            layer_norm: LayerNorm::load(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps as f32,
            )?,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;

        if let Some(position_embeddings) = &self.position_embeddings {
            let position_embeddings = position_embeddings.forward(position_ids)?;
            let embeddings = input_embeddings.add(&token_type_embeddings)?;
            self.layer_norm
                .forward(&embeddings, Some(&position_embeddings))
        } else {
            self.layer_norm
                .forward(&input_embeddings, Some(&token_type_embeddings))
        }
    }
}

pub trait ClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor>;

    // Token classification uses this hook in the CUDA model.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn forward_tokens(&self, hidden_states: &Tensor) -> Result<Tensor>;
}

pub struct JinaBertClassificationHead {
    pooler: Option<Linear>,
    output: Linear,
    span: tracing::Span,
}

impl JinaBertClassificationHead {
    pub(crate) fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let n_classes = match &config.id2label {
            None => candle::bail!("`id2label` must be set for classifier models"),
            Some(id2label) => id2label.len(),
        };

        let pooler = if let Ok(pooler_weight) = vb
            .pp("bert.pooler.dense")
            .get((config.hidden_size, config.hidden_size), "weight")
        {
            let pooler_bias = vb.pp("bert.pooler.dense").get(config.hidden_size, "bias")?;
            Some(Linear::new(pooler_weight, Some(pooler_bias), None))
        } else {
            None
        };

        let output_weight = vb
            .pp("classifier")
            .get((n_classes, config.hidden_size), "weight")?;
        let output_bias = vb.pp("classifier").get(n_classes, "bias")?;
        let output = Linear::new(output_weight, Some(output_bias), None);

        Ok(Self {
            pooler,
            output,
            span: tracing::span!(tracing::Level::TRACE, "classifier"),
        })
    }
}

impl ClassificationHead for JinaBertClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.unsqueeze(1)?;
        if let Some(pooler) = self.pooler.as_ref() {
            hidden_states = pooler.forward(&hidden_states)?;
            hidden_states = hidden_states.tanh()?;
        }

        let hidden_states = self.output.forward(&hidden_states)?;
        let hidden_states = hidden_states.squeeze(1)?;
        Ok(hidden_states)
    }

    fn forward_tokens(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.output.forward(hidden_states)?;
        Ok(hidden_states)
    }
}
