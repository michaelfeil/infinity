use crate::layers::{HiddenAct, LayerNorm, Linear};
use candle::{Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>,
    pub id2label: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
    Alibi,
    Rope,
}

#[derive(Debug)]
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertEmbeddings {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        if config.position_embedding_type != PositionEmbeddingType::Absolute {
            candle::bail!("Bert only supports absolute position embeddings");
        }

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
            position_embeddings: Embedding::new(
                vb.pp("position_embeddings").get(
                    (config.max_position_embeddings, config.hidden_size),
                    "weight",
                )?,
                config.hidden_size,
            ),
            layer_norm: LayerNorm::load(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps as f32,
            )?,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub(crate) fn word_embeddings_weight(&self) -> &Tensor {
        self.word_embeddings.embeddings()
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        // Packed CUDA inputs can use the existing vectorized row gather.
        // Preserve Embedding's shape handling for non-CUDA inputs.
        let gather = |embedding: &Embedding, ids: &Tensor| {
            if ids.rank() == 1 && matches!(ids.device(), Device::Cuda(_)) {
                crate::layers::index_select(embedding.embeddings(), ids, 0)
            } else {
                embedding.forward(ids)
            }
        };
        let input_embeddings = gather(&self.word_embeddings, input_ids)?;
        let token_type_embeddings = gather(&self.token_type_embeddings, token_type_ids)?;
        let position_embeddings = gather(&self.position_embeddings, position_ids)?;

        let embeddings = input_embeddings.add(&token_type_embeddings)?;
        let embeddings = self
            .layer_norm
            .forward(&embeddings, Some(&position_embeddings))?;

        Ok(embeddings)
    }
}

pub trait ClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor>;

    fn forward_tokens(&self, hidden_states: &Tensor) -> Result<Tensor>;
}

pub struct BertClassificationHead {
    pooler: Option<Linear>,
    output: Linear,
    span: tracing::Span,
}

impl BertClassificationHead {
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

impl ClassificationHead for BertClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        match &self.pooler {
            Some(pooler) => self.output.forward(&pooler.forward(hidden_states)?.tanh()?),
            None => self.output.forward(hidden_states),
        }
    }

    fn forward_tokens(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.output.forward(hidden_states)?;
        Ok(hidden_states)
    }
}

pub struct RobertaClassificationHead {
    intermediate: Linear,
    output: Linear,
    span: tracing::Span,
}

impl RobertaClassificationHead {
    pub(crate) fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let n_classes = match &config.id2label {
            None => candle::bail!("`id2label` must be set for classifier models"),
            Some(id2label) => id2label.len(),
        };

        let intermediate_weight = vb
            .pp("dense")
            .get((config.hidden_size, config.hidden_size), "weight")?;
        let intermediate_bias = vb.pp("dense").get(config.hidden_size, "bias")?;
        let intermediate = Linear::new(intermediate_weight, Some(intermediate_bias), None);

        let output_weight = vb
            .pp("out_proj")
            .get((n_classes, config.hidden_size), "weight")?;
        let output_bias = vb.pp("out_proj").get(n_classes, "bias")?;
        let output = Linear::new(output_weight, Some(output_bias), None);

        Ok(Self {
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "classifier"),
        })
    }
}

impl ClassificationHead for RobertaClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.intermediate.forward(hidden_states)?;
        let hidden_states = hidden_states.tanh()?;
        let hidden_states = self.output.forward(&hidden_states)?;
        Ok(hidden_states)
    }

    fn forward_tokens(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.intermediate.forward(hidden_states)?;
        let hidden_states = hidden_states.tanh()?;
        let hidden_states = self.output.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

pub(crate) fn load_roberta_classification_head(
    vb: VarBuilder,
    config: &BertConfig,
) -> Result<Box<dyn ClassificationHead + Send>> {
    if vb.contains_tensor("classifier.dense.weight") {
        Ok(Box::new(RobertaClassificationHead::load(
            vb.pp("classifier"),
            config,
        )?))
    } else {
        // RobertaForTokenClassification and XLMRobertaForTokenClassification use a plain
        // `classifier.{weight,bias}` head, unlike their sequence-classification counterparts.
        Ok(Box::new(BertClassificationHead::load(vb, config)?))
    }
}

#[derive(Debug)]
pub struct BertSpladeHead {
    transform: Linear,
    transform_layer_norm: LayerNorm,
    decoder: Linear,
    span: tracing::Span,
}

impl BertSpladeHead {
    pub(crate) fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let transform_weight = vb
            .pp("cls.predictions.transform.dense")
            .get((config.hidden_size, config.hidden_size), "weight")?;
        let transform_bias = vb
            .pp("cls.predictions.transform.dense")
            .get(config.hidden_size, "bias")?;
        let transform = Linear::new(
            transform_weight,
            Some(transform_bias),
            Some(config.hidden_act.clone()),
        );

        let transform_layer_norm = LayerNorm::load(
            vb.pp("cls.predictions.transform.LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        // When `pytorch_model.bin` originally contains `cls.predictions.decoder.weight` but the
        // tensor content shares the memory with the content on `bert.embeddings.word_embeddings.weight`,
        // e.g. a subset of the original tensor, when converting the file from BIN to Safentensors
        // the latter tensor that shares the memory with the previous will be removed
        let decoder_weight = if vb.contains_tensor("cls.predictions.decoder.weight") {
            vb.pp("cls.predictions.decoder")
                .get((config.vocab_size, config.hidden_size), "weight")?
        } else {
            vb.pp("bert.embeddings.word_embeddings")
                .get((config.vocab_size, config.hidden_size), "weight")?
        };
        // Same applies for the tensor `cls.predictions.decoder.bias` which is shared with
        // `cls.predictions.bias` and removed in the BIN to Safentensors conversion
        let decoder_bias = vb.pp("cls.predictions").get(config.vocab_size, "bias")?;
        let decoder = Linear::new(decoder_weight, Some(decoder_bias), Some(HiddenAct::Relu));

        Ok(Self {
            transform,
            transform_layer_norm,
            decoder,
            span: tracing::span!(tracing::Level::TRACE, "splade"),
        })
    }

    pub(crate) fn load_roberta(
        vb: VarBuilder,
        config: &BertConfig,
        tied_embeddings: &Tensor,
    ) -> Result<Self> {
        let vb = vb.pp("lm_head");
        let transform_weight = vb
            .pp("dense")
            .get((config.hidden_size, config.hidden_size), "weight")?;
        let transform_bias = vb.pp("dense").get(config.hidden_size, "bias")?;
        let transform = Linear::new(
            transform_weight,
            Some(transform_bias),
            Some(HiddenAct::Gelu),
        );

        let transform_layer_norm = LayerNorm::load(
            vb.pp("layer_norm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        // Safetensors may omit the decoder tied to the input embeddings, as
        // with the original RoBERTa masked-language-model checkpoint. Use the
        // embedding weight from the backbone that actually loaded, regardless
        // of its checkpoint namespace.
        let decoder_weight = if vb.contains_tensor("decoder.weight") {
            vb.pp("decoder")
                .get((config.vocab_size, config.hidden_size), "weight")?
        } else {
            tied_embeddings.clone()
        };
        let decoder_bias = vb.get(config.vocab_size, "bias")?;
        let decoder = Linear::new(decoder_weight, Some(decoder_bias), Some(HiddenAct::Relu));

        Ok(Self {
            transform,
            transform_layer_norm,
            decoder,
            span: tracing::span!(tracing::Level::TRACE, "splade"),
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.transform.forward(hidden_states)?;
        let hidden_states = self.transform_layer_norm.forward(&hidden_states, None)?;
        let hidden_states = self.decoder.forward(&hidden_states)?;
        (1.0 + hidden_states)?.log()
    }
}

#[cfg(test)]
mod classification_tests {
    use super::*;

    #[test]
    fn roberta_splade_tied_decoder_fallback_preserves_explicit_weights() -> Result<()> {
        let device = Device::Cpu;
        let config: BertConfig = serde_json::from_value(serde_json::json!({
            "vocab_size": 3, "hidden_size": 2, "num_hidden_layers": 1,
            "num_attention_heads": 1, "intermediate_size": 4,
            "hidden_act": "gelu", "hidden_dropout_prob": 0.,
            "max_position_embeddings": 4, "type_vocab_size": 1,
            "initializer_range": 0.02, "layer_norm_eps": 1e-5, "pad_token_id": 1
        }))
        .unwrap();
        let embeddings = Tensor::new(&[[1_f32, -1.], [-1., 1.], [0.5, 0.25]], &device)?;
        let mut tensors = HashMap::from([
            (
                "lm_head.dense.weight".into(),
                Tensor::new(&[[1_f32, 0.], [0., 1.]], &device)?,
            ),
            (
                "lm_head.dense.bias".into(),
                Tensor::zeros(2, candle::DType::F32, &device)?,
            ),
            (
                "lm_head.layer_norm.weight".into(),
                Tensor::ones(2, candle::DType::F32, &device)?,
            ),
            (
                "lm_head.layer_norm.bias".into(),
                Tensor::zeros(2, candle::DType::F32, &device)?,
            ),
            (
                "lm_head.bias".into(),
                Tensor::new(&[0.1_f32, 0.2, 0.3], &device)?,
            ),
            (
                "roberta.embeddings.word_embeddings.weight".into(),
                // An unrelated partial namespace must not supply the tied decoder.
                Tensor::zeros((3, 2), candle::DType::F32, &device)?,
            ),
        ]);
        let input = Tensor::new(&[[0.5_f32, -1.], [-0.2, 0.7]], &device)?;
        let tied = BertSpladeHead::load_roberta(
            VarBuilder::from_tensors(tensors.clone(), candle::DType::F32, &device),
            &config,
            &embeddings,
        )?
        .forward(&input)?
        .to_vec2::<f32>()?;
        tensors.insert("lm_head.decoder.weight".into(), embeddings.clone());
        let explicit_tied = BertSpladeHead::load_roberta(
            VarBuilder::from_tensors(tensors.clone(), candle::DType::F32, &device),
            &config,
            &embeddings,
        )?
        .forward(&input)?
        .to_vec2::<f32>()?;
        assert_eq!(tied, explicit_tied);
        tensors.insert(
            "lm_head.decoder.weight".into(),
            Tensor::zeros((3, 2), candle::DType::F32, &device)?,
        );
        let explicit = BertSpladeHead::load_roberta(
            VarBuilder::from_tensors(tensors, candle::DType::F32, &device),
            &config,
            &embeddings,
        )?
        .forward(&input)?
        .to_vec2::<f32>()?;
        assert_ne!(tied, explicit);
        for row in explicit {
            for (&actual, bias) in row.iter().zip([0.1_f32, 0.2, 0.3]) {
                assert!((actual - (1. + bias).ln()).abs() < 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn classification_head_batch_matches_independent_rows() -> Result<()> {
        let device = Device::Cpu;
        let head = BertClassificationHead {
            pooler: Some(Linear::new(
                Tensor::new(&[[1_f32, 2.], [3., 4.]], &device)?,
                None,
                None,
            )),
            output: Linear::new(
                Tensor::new(&[[0.4_f32, -0.3], [0.2, 0.7]], &device)?,
                None,
                None,
            ),
            span: tracing::Span::none(),
        };
        let input = Tensor::new(&[[0.5_f32, -1.], [1., 0.2]], &device)?;
        let batch = head.forward(&input)?.to_vec2::<f32>()?;
        for (i, row) in batch.iter().enumerate() {
            let single = head.forward(&input.narrow(0, i, 1)?)?.to_vec2::<f32>()?;
            for (&actual, &expected) in row.iter().zip(&single[0]) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
        Ok(())
    }
    #[test]
    fn roberta_classification_batch_matches_independent_rows() -> Result<()> {
        let device = Device::Cpu;
        let head = RobertaClassificationHead {
            intermediate: Linear::new(
                Tensor::new(&[[1_f32, 2.], [3., 4.]], &device)?,
                Some(Tensor::new(&[0.1_f32, -0.2], &device)?),
                None,
            ),
            output: Linear::new(
                Tensor::new(&[[0.4_f32, -0.3], [0.2, 0.7]], &device)?,
                None,
                None,
            ),
            span: tracing::Span::none(),
        };
        let input = Tensor::new(&[[0.5_f32, -1.], [1., 0.2]], &device)?;
        let batch = head.forward(&input)?.to_vec2::<f32>()?;
        for (i, row) in batch.iter().enumerate() {
            let single = head.forward(&input.narrow(0, i, 1)?)?.to_vec2::<f32>()?;
            for (&actual, &expected) in row.iter().zip(&single[0]) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
        Ok(())
    }
}
