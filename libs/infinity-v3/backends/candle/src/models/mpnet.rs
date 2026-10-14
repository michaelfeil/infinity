use crate::layers::HiddenAct;
use serde::Deserialize;

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpnet/configuration_mpnet.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MPNetConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub relative_attention_num_buckets: usize,
}
