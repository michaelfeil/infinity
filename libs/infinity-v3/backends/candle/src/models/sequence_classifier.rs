//! Hugging Face decoder sequence classification: final non-padding token + score head.
use super::Model;
use crate::layers::{index_select, Linear};
use candle::{Result, Tensor};
use candle_nn::VarBuilder;
use infinity_v3_backend_core::Batch;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Config {
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: usize,
    num_labels: Option<usize>,
    id2label: Option<HashMap<String, String>>,
    pad_token_id: Option<u32>,
}

pub(crate) struct SequenceClassifier {
    model: Box<dyn Model + Send>,
    head: Linear,
    pad_token_id: Option<u32>,
}

impl SequenceClassifier {
    pub(crate) fn supports(config: &str) -> Result<bool> {
        let config: Config = serde_json::from_str(config).map_err(candle::Error::wrap)?;
        Ok(
            matches!(config.architectures.as_slice(), [architecture] if matches!(architecture.as_str(),
                "LlamaForSequenceClassification" | "Qwen2ForSequenceClassification" | "Qwen3ForSequenceClassification"
                    | "Qwen3MoeForSequenceClassification"
            )),
        )
    }

    pub(crate) fn load(model: Box<dyn Model + Send>, vb: VarBuilder, config: &str) -> Result<Self> {
        let config: Config = serde_json::from_str(config).map_err(candle::Error::wrap)?;
        let labels = config
            .id2label
            .as_ref()
            .map(HashMap::len)
            .or(config.num_labels)
            .unwrap_or(2);
        if labels == 0 || config.num_labels.is_some_and(|n| n != labels) {
            candle::bail!("Sequence classifier num_labels and id2label must agree and be nonzero");
        }
        let weight = vb.pp("score").get((labels, config.hidden_size), "weight")?;
        let bias = if vb.contains_tensor("score.bias") {
            Some(vb.pp("score").get(labels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            model,
            head: Linear::new(weight, bias, None),
            pad_token_id: config.pad_token_id,
        })
    }
}

impl Model for SequenceClassifier {
    fn supports_radix_mlp(&self) -> bool {
        self.model.supports_radix_mlp()
    }

    fn predict(&self, mut batch: Batch) -> Result<Tensor> {
        let mut indices = Vec::with_capacity(batch.len());
        for range in batch.cumulative_seq_lengths.windows(2) {
            let (start, end) = (range[0] as usize, range[1] as usize);
            if start == end {
                candle::bail!("Sequence classification requires nonempty inputs");
            }
            // HF selects the final non-pad ID, including when EOS doubles as padding.
            // Match HF's first-token fallback when every token is padding.
            let index = match self.pad_token_id {
                Some(pad) => batch.input_ids[start..end]
                    .iter()
                    .rposition(|&id| id != pad)
                    .unwrap_or(0),
                None => end - start - 1,
            };
            indices.push((start + index) as u32);
        }
        // Reuse the model's packed execution and Radix unfold. Gather before
        // applying score.weight, avoiding a classifier projection for every token.
        batch.raw_indices = (0..batch.len() as u32).collect();
        batch.pooled_indices.clear();
        let (_, hidden) = self.model.embed(batch)?;
        let hidden = hidden.ok_or_else(|| {
            candle::Error::Msg("Classifier backbone returned no token states".into())
        })?;
        let indices = Tensor::new(indices, hidden.device())?;
        self.head.forward(&index_select(&hidden, &indices, 0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    struct TokenStates;
    impl Model for TokenStates {
        fn supports_radix_mlp(&self) -> bool {
            true
        }
        fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
            assert!(batch.pooled_indices.is_empty());
            assert_eq!(batch.raw_indices, vec![0, 1, 2]);
            let n = batch.input_ids.len();
            let states: Vec<f32> = batch.input_ids.into_iter().map(|id| id as f32).collect();
            Ok((None, Some(Tensor::from_vec(states, (n, 1), &Device::Cpu)?)))
        }
    }

    #[test]
    fn only_sequence_classification_architectures_are_supported() -> Result<()> {
        for family in ["Llama", "Qwen2", "Qwen3", "Qwen3Moe"] {
            for (suffix, supported) in [
                ("ForSequenceClassification", true),
                ("ForTokenClassification", false),
                ("ForCausalLM", false),
            ] {
                let config = serde_json::json!({"hidden_size":16,"architectures":[format!("{family}{suffix}")]}).to_string();
                assert_eq!(SequenceClassifier::supports(&config)?, supported);
            }
        }
        assert!(!SequenceClassifier::supports(r#"{"hidden_size":16}"#)?);
        assert!(!SequenceClassifier::supports(
            r#"{"hidden_size":16,"architectures":["Qwen3ForSequenceClassification","Qwen3ForTokenClassification"]}"#
        )?);
        Ok(())
    }

    #[test]
    fn last_non_pad_pooling_preserves_order_and_radix_capability() -> Result<()> {
        let vb = VarBuilder::from_tensors(
            HashMap::from([(
                "score.weight".into(),
                Tensor::new(&[[1f32], [-1.]], &Device::Cpu)?,
            )]),
            DType::F32,
            &Device::Cpu,
        );
        let model = SequenceClassifier::load(
            Box::new(TokenStates),
            vb,
            r#"{"hidden_size":1,"id2label":{"0":"no","1":"yes"},"pad_token_id":0}"#,
        )?;
        assert!(model.supports_radix_mlp());
        let batch = Batch {
            multimodal: vec![],
            input_ids: vec![1, 2, 0, 0, 3, 0, 4, 0, 0],
            token_type_ids: vec![0; 9],
            position_ids: vec![0, 1, 2, 3, 0, 1, 2, 0, 1],
            cumulative_seq_lengths: vec![0, 4, 7, 9],
            max_length: 4,
            pooled_indices: vec![0, 1, 2],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        assert_eq!(
            model.predict(batch)?.to_vec2::<f32>()?,
            vec![vec![2., -2.], vec![4., -4.], vec![0., 0.]]
        );
        Ok(())
    }

    #[test]
    fn rejects_missing_or_mismatched_score_head() -> Result<()> {
        let empty = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
        assert!(SequenceClassifier::load(
            Box::new(TokenStates),
            empty,
            r#"{"hidden_size":1,"num_labels":2}"#
        )
        .is_err());
        let vb = VarBuilder::from_tensors(
            HashMap::from([("score.weight".into(), Tensor::new(&[[1f32]], &Device::Cpu)?)]),
            DType::F32,
            &Device::Cpu,
        );
        assert!(SequenceClassifier::load(
            Box::new(TokenStates),
            vb.clone(),
            r#"{"hidden_size":1,"num_labels":2}"#
        )
        .is_err());
        assert!(SequenceClassifier::load(
            Box::new(TokenStates),
            vb,
            r#"{"hidden_size":1,"num_labels":2,"id2label":{"0":"x"}}"#
        )
        .is_err());
        Ok(())
    }
}
