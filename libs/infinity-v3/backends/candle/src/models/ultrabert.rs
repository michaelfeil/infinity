use std::collections::HashMap;

use crate::layers::{
    get_cos_sin, get_inv_freqs, index_select, CompactUnfoldTensors, HiddenAct, RMSNorm,
};
use crate::models::{flash_mistral::MistralLayer, MistralConfig, Model};
use candle::{Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use infinity_v3_backend_core::{Batch, ModelType, Pool};
use serde::Deserialize;

// The released checkpoint omits a position limit; its tokenizer advertises 32K.
pub(crate) const MAX_POSITION_EMBEDDINGS: usize = 32768;

#[derive(Debug, Deserialize)]
pub struct UltraBertConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_act: HiddenAct,
    rms_norm_eps: f32,
    rope_theta: Vec<f32>,
    attention_windows: Vec<Option<usize>>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    architectures: Vec<String>,
    num_labels: Option<usize>,
    id2label: Option<HashMap<String, String>>,
}

impl UltraBertConfig {
    fn layer_config(&self, index: usize) -> MistralConfig {
        MistralConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: Some(self.head_dim),
            hidden_act: self.hidden_act.clone(),
            max_position_embeddings: self
                .max_position_embeddings
                .unwrap_or(MAX_POSITION_EMBEDDINGS),
            initializer_range: 0.02,
            rms_norm_eps: self.rms_norm_eps,
            model_type: Some("ultrabert".into()),
            rope_theta: self.rope_theta[index],
            // UltraBERT's local mask includes both endpoints at this distance.
            sliding_window: self.attention_windows[index],
            rope_scaling: None,
            use_bidirectional_attention: true,
            is_causal: Some(false),
            rope_parameters: None,
        }
    }
}

pub struct UltraBertModel {
    embeddings: Embedding,
    layers: Vec<MistralLayer>,
    norm: RMSNorm,
    rotary: Vec<(Tensor, Tensor)>,
    pool: Pool,
    device: Device,
    classifier: Option<(crate::layers::Linear, bool)>,
}

impl UltraBertModel {
    pub fn load(vb: VarBuilder, config: &UltraBertConfig, model_type: ModelType) -> Result<Self> {
        crate::flash_attn::validate_packed_device(&vb)?;
        if config.attention_bias || config.mlp_bias {
            candle::bail!("UltraBERT requires bias-free attention and MLP projections");
        }
        if config.num_hidden_layers == 0
            || config.num_attention_heads == 0
            || config.num_key_value_heads == 0
            || !config
                .num_attention_heads
                .is_multiple_of(config.num_key_value_heads)
            || config.head_dim == 0
            || !config.head_dim.is_multiple_of(2)
            || config.rope_theta.len() != config.num_hidden_layers
            || config.attention_windows.len() != config.num_hidden_layers
            || config
                .rope_theta
                .iter()
                .any(|theta| !theta.is_finite() || *theta <= 0.)
            || config.max_position_embeddings == Some(0)
        {
            candle::bail!("Invalid UltraBERT head geometry or per-layer attention configuration");
        }
        let (pool, classifier) = match model_type {
            ModelType::Embedding(pool @ (Pool::Cls | Pool::Mean | Pool::LastToken)) => (pool, None),
            ModelType::Classifier => {
                let token_classifier = match config.architectures.as_slice() {
                    [arch] if arch == "UltrabertForSequenceClassification" => false,
                    [arch] if arch == "UltrabertForTokenClassification" => true,
                    _ => candle::bail!("UltraBERT classification requires a sequence or token classification checkpoint"),
                };
                let labels = config
                    .id2label
                    .as_ref()
                    .map(HashMap::len)
                    .or(config.num_labels)
                    .unwrap_or(2);
                if labels == 0 || config.num_labels.is_some_and(|n| n != labels) {
                    candle::bail!("UltraBERT num_labels and id2label must agree and be nonzero");
                }
                let head = crate::layers::Linear::new(
                    vb.pp("classifier")
                        .get((labels, config.hidden_size), "weight")?,
                    Some(vb.pp("classifier").get(labels, "bias")?),
                    None,
                );
                (Pool::Cls, Some((head, token_classifier)))
            }
            _ => candle::bail!("UltraBERT supports embeddings and sequence/token classification"),
        };
        let vb = if vb.contains_tensor("ultrabert.embed_tokens.weight") {
            vb.pp("ultrabert")
        } else {
            vb
        };
        let embeddings = Embedding::new(
            vb.pp("embed_tokens")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut rotary = Vec::with_capacity(config.num_hidden_layers);
        let mut caches = HashMap::new();
        for index in 0..config.num_hidden_layers {
            let layer_config = config.layer_config(index);
            layers.push(
                MistralLayer::load(vb.pp(format!("layers.{index}")), &layer_config, false)?
                    .with_symmetric_window(),
            );
            let key = layer_config.rope_theta.to_bits();
            if let std::collections::hash_map::Entry::Vacant(entry) = caches.entry(key) {
                let freqs =
                    get_inv_freqs(config.head_dim, layer_config.rope_theta, vb.device(), None)?;
                entry.insert(get_cos_sin(
                    layer_config.max_position_embeddings,
                    &freqs,
                    vb.dtype(),
                    false,
                )?);
            }
            rotary.push(caches[&key].clone());
        }
        Ok(Self {
            embeddings,
            layers,
            norm: RMSNorm::load(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?,
            rotary,
            pool,
            device: vb.device().clone(),
            classifier,
        })
    }
}

impl UltraBertModel {
    fn forward_hidden(&self, batch: &Batch) -> Result<Tensor> {
        let (ids, compact) = CompactUnfoldTensors::from_batch(batch, &self.device)?;
        let mut hidden = self.embeddings.forward(&ids)?.contiguous()?;
        let offsets = Tensor::new(batch.cumulative_seq_lengths.as_slice(), &self.device)?;
        #[cfg(feature = "fa4")]
        let _fa4_batch = crate::fa4_native::prepare_batch(&offsets, &batch.cumulative_seq_lengths)?;
        let mut residual = None;
        let mut positions = HashMap::new();
        for (layer, (cos, sin)) in self.layers.iter().zip(&self.rotary) {
            let key = cos.id();
            if let std::collections::hash_map::Entry::Vacant(entry) = positions.entry(key) {
                entry.insert((
                    index_select(cos, &compact.position_ids_compact, 0)?,
                    index_select(sin, &compact.position_ids_compact, 0)?,
                ));
            }
            let (cos, sin) = &positions[&key];
            let (h, r) = layer.forward(
                &hidden,
                residual.as_ref(),
                &offsets,
                cos,
                sin,
                None,
                batch.max_length as usize,
                &compact,
            )?;
            hidden = h;
            residual = Some(r);
        }
        let (hidden, _) = self.norm.forward(&hidden, residual.as_ref())?;
        compact.scatter_unfold(&hidden)
    }
}

impl Model for UltraBertModel {
    fn predict(&self, batch: Batch) -> Result<Tensor> {
        let Some((head, _)) = &self.classifier else {
            candle::bail!("UltraBERT prediction requires a classification checkpoint");
        };
        // Generic classifier health/warmup uses this path for token heads too.
        let states = self.forward_hidden(&batch)?;
        let indices = Tensor::new(&batch.cumulative_seq_lengths[..batch.len()], &self.device)?;
        head.forward(&index_select(&states, &indices, 0)?)
    }

    fn predict_tokens(&self, batch: Batch) -> Result<Tensor> {
        let Some((head, true)) = &self.classifier else {
            candle::bail!("UltraBERT token prediction requires a token classification checkpoint");
        };
        head.forward(&self.forward_hidden(&batch)?)
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if self.classifier.is_some() {
            candle::bail!("UltraBERT embedding requests require an embedding checkpoint");
        }
        let outputs = self.forward_hidden(&batch)?;
        let select = |indices: Vec<u32>| -> Result<Tensor> {
            index_select(&outputs, &Tensor::new(indices, &self.device)?, 0)
        };
        let pooled = if batch.pooled_indices.is_empty() {
            None
        } else if self.pool == Pool::Mean {
            Some(crate::layers::mean_pool(
                &outputs,
                &batch.cumulative_seq_lengths,
                &batch.pooled_indices,
            )?)
        } else {
            Some(select(
                batch
                    .pooled_indices
                    .iter()
                    .map(|&index| {
                        let index = index as usize;
                        if self.pool == Pool::Cls {
                            batch.cumulative_seq_lengths[index]
                        } else {
                            batch.cumulative_seq_lengths[index + 1] - 1
                        }
                    })
                    .collect(),
            )?)
        };
        let raw = if batch.raw_indices.is_empty() {
            None
        } else {
            Some(select(
                batch
                    .raw_indices
                    .iter()
                    .flat_map(|&index| {
                        let index = index as usize;
                        batch.cumulative_seq_lengths[index]..batch.cumulative_seq_lengths[index + 1]
                    })
                    .collect(),
            )?)
        };
        Ok((pooled, raw))
    }
}

#[cfg(test)]
mod classification_tests {
    use super::*;
    use candle::DType;

    #[test]
    fn classifiers_require_consistent_labels_and_saved_heads() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &device);
        let mut value = serde_json::json!({
            "vocab_size":64,"hidden_size":16,"intermediate_size":24,"num_hidden_layers":1,
            "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,"hidden_act":"silu",
            "rms_norm_eps":1e-6,"rope_theta":[10000.],"attention_windows":[null],
            "architectures":["UltrabertForSequenceClassification"],"num_labels":2,
            "id2label":{"0":"negative","1":"positive"}
        });
        let config = serde_json::from_value(value.clone()).unwrap();
        let error = UltraBertModel::load(vb.clone(), &config, ModelType::Classifier)
            .err()
            .unwrap();
        assert!(error.to_string().contains("classifier.weight"));
        for labels in [0, 3] {
            value["num_labels"] = labels.into();
            let config = serde_json::from_value(value.clone()).unwrap();
            let error = UltraBertModel::load(vb.clone(), &config, ModelType::Classifier)
                .err()
                .unwrap();
            assert!(error.to_string().contains("num_labels"));
        }
        Ok(())
    }
}
