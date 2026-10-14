use crate::layers::{HiddenAct, Linear, RopeScaling};
use crate::models::PositionEmbeddingType;
use candle::{Result, Tensor, D};
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GTEConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_type: String,
    pub layer_norm_eps: f32,
    pub position_embedding_type: PositionEmbeddingType,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub logn_attention_scale: bool,
    #[serde(default)]
    pub logn_attention_clip1: bool,
    pub id2label: Option<HashMap<String, String>>,
}

#[allow(clippy::upper_case_acronyms)]
pub struct GTEMLP {
    up_gate_proj: Linear,
    down_proj: Linear,

    act: HiddenAct,
    intermediate_size: usize,

    span: tracing::Span,
}

impl GTEMLP {
    pub fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let intermediate_size = config.intermediate_size;

        let up_gate_proj_weight = vb
            .pp("up_gate_proj")
            .get((intermediate_size * 2, config.hidden_size), "weight")?;
        let up_gate_proj = Linear::new(up_gate_proj_weight, None, None);

        let down_proj_weight = vb
            .pp("down_proj")
            .get((config.hidden_size, intermediate_size), "weight")?;
        let down_proj_bias = vb.pp("down_proj").get(config.hidden_size, "bias")?;
        let down_proj = Linear::new(down_proj_weight, Some(down_proj_bias), None);

        Ok(Self {
            up_gate_proj,
            down_proj,
            intermediate_size,
            act: config.hidden_act.clone(),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let up_gate_states = self.up_gate_proj.forward(hidden_states)?;
        let up_states = up_gate_states.narrow(D::Minus1, 0, self.intermediate_size)?;

        let gate =
            up_gate_states.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let gate = self.act.forward(&gate)?;

        self.down_proj.forward(&(gate * up_states)?)
    }
}

pub struct GTEClassificationHead {
    pooler: Option<Linear>,
    classifier: Linear,
    span: tracing::Span,
}

impl GTEClassificationHead {
    fn inner_load(vb: VarBuilder, config: &GTEConfig) -> Option<Linear> {
        let pooler_weight = vb
            .pp("pooler.dense")
            .get((config.hidden_size, config.hidden_size), "weight")
            .ok()?;
        let pooler_bias = vb.pp("pooler.dense").get(config.hidden_size, "bias").ok()?;
        let pooler = Linear::new(pooler_weight, Some(pooler_bias), None);

        Some(pooler)
    }

    pub(crate) fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let n_classes = match &config.id2label {
            None => candle::bail!("`id2label` must be set for classifier models"),
            Some(id2label) => id2label.len(),
        };

        let pooler =
            Self::inner_load(vb.pp("new"), config).or_else(|| Self::inner_load(vb.clone(), config));

        let classifier_weight = vb
            .pp("classifier")
            .get((n_classes, config.hidden_size), "weight")?;
        let classifier_bias = vb.pp("classifier").get(n_classes, "bias")?;
        let classifier = Linear::new(classifier_weight, Some(classifier_bias), None);

        Ok(Self {
            classifier,
            pooler,
            span: tracing::span!(tracing::Level::TRACE, "classifier"),
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.unsqueeze(1)?;

        if let Some(pooler) = self.pooler.as_ref() {
            hidden_states = pooler.forward(&hidden_states)?;
            hidden_states = hidden_states.tanh()?;
        }

        let hidden_states = self.classifier.forward(&hidden_states)?;
        let hidden_states = hidden_states.squeeze(1)?;

        Ok(hidden_states)
    }
}
