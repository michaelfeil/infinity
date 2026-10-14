//! Laya's bidirectional ModernBERT encoder and typed decision head.
//!
//! The checkpoint is not a sequence classifier: each `[MASK]` marker names an
//! option, and its trained head scores all markers in one encoded question.
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};
use infinity_v3_backend_core::{Batch, ModelType, Pool};
use serde::Deserialize;

use crate::layers::{LayerNorm, Linear};
use crate::models::{FlashModernBertModel, ModernBertConfig};

#[derive(Debug, Clone, Deserialize)]
pub struct LayaConfig {
    pub encoder: String,
    pub head_layers: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    #[serde(default)]
    pub temperature: Vec<f32>,
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f32>,
}

impl LayaConfig {
    /// Read the checkpoint's own configuration and its embedded encoder config.
    fn from_model_dir(path: &Path) -> anyhow::Result<(Self, ModernBertConfig)> {
        let cfg: Self = serde_json::from_slice(&fs::read(path.join("rl_agent_config.json"))?)?;
        anyhow::ensure!(
            cfg.head_layers <= 16,
            "Laya head_layers exceeds the supported limit"
        );
        anyhow::ensure!(
            cfg.max_len > 0 && cfg.head_max_len > 0,
            "Laya token budgets must be positive"
        );
        let mut encoder: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("encoder/config.json"))?)?;
        anyhow::ensure!(
            encoder["model_type"] == "modernbert",
            "Laya encoder must be ModernBERT"
        );
        // Transformers 5 stores the two RoPE bases in rope_parameters; infinity_v3's
        // ModernBERT config still consumes the legacy flat names.
        for (kind, field) in [
            ("full_attention", "global_rope_theta"),
            ("sliding_attention", "local_rope_theta"),
        ] {
            if encoder.get(field).is_none() {
                let theta = encoder["rope_parameters"][kind]["rope_theta"]
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("missing {kind} rope_theta"))?;
                encoder[field] = serde_json::json!(theta);
            }
        }
        let encoder = serde_json::from_value(encoder)?;
        Ok((cfg, encoder))
    }

    pub fn temperature_for(&self, question_type: usize, options: usize) -> f32 {
        let kind = ["choice", "score", "noul"]
            .get(question_type)
            .copied()
            .unwrap_or("choice");
        let bucket = if options <= 2 {
            "2"
        } else if options <= 5 {
            "3-5"
        } else if options <= 10 {
            "6-10"
        } else {
            "11+"
        };
        let key = format!("{kind}:{bucket}");
        let value = self
            .temperature_by_options
            .get(&key)
            .copied()
            .or_else(|| self.temperature.get(question_type).copied())
            .unwrap_or(1.0);
        if value.is_finite() {
            value.clamp(0.5, 5.0)
        } else {
            1.0
        }
    }
}

struct HeadLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    out: Linear,
    linear1: Linear,
    linear2: Linear,
    heads: usize,
    head_size: usize,
}

impl HeadLayer {
    fn load(vb: VarBuilder, hidden: usize) -> Result<Self> {
        let linear = |vb: VarBuilder, input: usize, output: usize| -> Result<Linear> {
            Ok(Linear::new(
                vb.get((output, input), "weight")?,
                Some(vb.get(output, "bias")?),
                None,
            ))
        };
        let heads = (hidden / 64).max(1);
        if !hidden.is_multiple_of(heads) {
            candle::bail!("Laya hidden size must be divisible by head count")
        }
        Ok(Self {
            norm1: LayerNorm::load(vb.pp("norm1"), hidden, 1e-5)?,
            norm2: LayerNorm::load(vb.pp("norm2"), hidden, 1e-5)?,
            qkv: Linear::new(
                vb.pp("self_attn")
                    .get((3 * hidden, hidden), "in_proj_weight")?,
                Some(vb.pp("self_attn").get(3 * hidden, "in_proj_bias")?),
                None,
            ),
            out: linear(vb.pp("self_attn.out_proj"), hidden, hidden)?,
            linear1: linear(vb.pp("linear1"), hidden, 4 * hidden)?,
            linear2: linear(vb.pp("linear2"), 4 * hidden, hidden)?,
            heads,
            head_size: hidden / heads,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        // PyTorch TransformerEncoderLayer(norm_first=True), in inference mode.
        let (_, length, width) = hidden.dims3()?;
        let normed = self.norm1.forward(hidden, None)?;
        let qkv = self
            .qkv
            .forward(&normed)?
            .reshape((1, length, 3, self.heads, self.head_size))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.contiguous()?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;
        let scores = (q.matmul(&k.transpose(2, 3)?)? / (self.head_size as f64).sqrt())?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((1, length, width))?;
        let hidden = (hidden + self.out.forward(&attn)?)?;
        let ff = self
            .linear1
            .forward(&self.norm2.forward(&hidden, None)?)?
            .relu()?;
        hidden + self.linear2.forward(&ff)?
    }
}

#[derive(Debug, Clone)]
pub struct LayaOutput {
    pub logits: Vec<f32>,
    pub action_probability: f32,
}

pub struct LayaModel {
    encoder: FlashModernBertModel,
    type_emb: Embedding,
    head: Vec<HeadLayer>,
    scorer_norm: LayerNorm,
    scorer_in: Linear,
    scorer_out: Linear,
    action_in: Linear,
    action_out: Linear,
    device: Device,
    dtype: DType,
    hidden: usize,
}

impl LayaModel {
    /// Load the published Laya checkpoint directly from a Hub snapshot directory.
    pub fn from_model_dir(
        path: &Path,
        dtype: DType,
        device: &Device,
        enable_fp8_dynamic: bool,
    ) -> anyhow::Result<(Self, LayaConfig)> {
        let (config, encoder_config) = LayaConfig::from_model_dir(path)?;
        let weights = path.join("model.safetensors");
        anyhow::ensure!(weights.is_file(), "Laya model.safetensors is missing");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let model = Self::load(vb, &encoder_config, &config, enable_fp8_dynamic)?;
        Ok((model, config))
    }

    fn load(
        vb: VarBuilder,
        encoder_config: &ModernBertConfig,
        config: &LayaConfig,
        enable_fp8_dynamic: bool,
    ) -> Result<Self> {
        let hidden = encoder_config.hidden_size;
        let linear = |vb: VarBuilder, input: usize, output: usize| -> Result<Linear> {
            Ok(Linear::new(
                vb.get((output, input), "weight")?,
                Some(vb.get(output, "bias")?),
                None,
            ))
        };
        let head = (0..config.head_layers)
            .map(|i| HeadLayer::load(vb.pp(format!("head.layers.{i}")), hidden))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            encoder: FlashModernBertModel::load(
                vb.pp("encoder"),
                encoder_config,
                ModelType::Embedding(Pool::Cls),
                enable_fp8_dynamic,
            )?,
            type_emb: Embedding::new(vb.pp("type_emb").get((3, hidden), "weight")?, hidden),
            head,
            scorer_norm: LayerNorm::load(vb.pp("scorer.0"), hidden, 1e-5)?,
            scorer_in: linear(vb.pp("scorer.1"), hidden, hidden)?,
            scorer_out: linear(vb.pp("scorer.3"), hidden, 1)?,
            action_in: linear(vb.pp("act_head.0"), hidden + 4, 256)?,
            action_out: linear(vb.pp("act_head.2"), 256, 2)?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden,
        })
    }

    /// Score one typed question. `markers` point at each option's `[MASK]` token.
    /// The caller owns tokenization, validation, and probability calibration.
    pub fn forward(
        &self,
        ids: &[u32],
        markers: &[usize],
        question_type: usize,
    ) -> Result<LayaOutput> {
        if ids.is_empty()
            || markers.is_empty()
            || question_type > 2
            || markers.iter().any(|&p| p >= ids.len())
        {
            candle::bail!("invalid Laya decision input")
        }
        let length = ids.len();
        let batch = Batch {
            multimodal: vec![],
            input_ids: ids.to_vec(),
            token_type_ids: vec![0; length],
            position_ids: (0..length as u32).collect(),
            cumulative_seq_lengths: vec![0, length as u32],
            max_length: length as u32,
            pooled_indices: vec![],
            raw_indices: vec![0],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let (_, raw) = self.encoder.forward(batch)?;
        let hidden = raw
            .ok_or_else(|| candle::Error::Msg("Laya encoder returned no hidden states".into()))?
            .reshape((1, length, self.hidden))?;
        self.forward_head(hidden, markers, question_type)
    }

    fn forward_head(
        &self,
        hidden: Tensor,
        markers: &[usize],
        question_type: usize,
    ) -> Result<LayaOutput> {
        let type_id = Tensor::from_vec(vec![question_type as u32], (1,), &self.device)?;
        let type_vector = self
            .type_emb
            .forward(&type_id)?
            .reshape((1, 1, self.hidden))?;
        let mut hidden = hidden.broadcast_add(&type_vector)?;
        for layer in &self.head {
            hidden = layer.forward(&hidden)?;
        }
        let marker_ids = Tensor::from_vec(
            markers.iter().map(|&p| p as u32).collect(),
            markers.len(),
            &self.device,
        )?;
        let selected = hidden.i(0)?.index_select(&marker_ids, 0)?;
        let logits = self
            .scorer_out
            .forward(
                &self
                    .scorer_in
                    .forward(&self.scorer_norm.forward(&selected, None)?)?
                    .gelu_erf()?,
            )?
            .to_dtype(DType::F32)?
            .reshape(markers.len())?
            .to_vec1::<f32>()?;

        let mut p = logits.clone();
        let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in &mut p {
            *v = (*v - max).exp();
        }
        let total: f32 = p.iter().sum();
        for v in &mut p {
            *v /= total;
        }
        let mut sorted = p.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let top1 = sorted[0];
        let top2 = sorted.get(1).copied().unwrap_or(0.0);
        let entropy =
            -p.iter().map(|&v| v * v.max(1e-9).ln()).sum::<f32>() / (p.len().max(2) as f32).ln();
        let features = [top1, top1 - top2, entropy, p.len().max(2) as f32 / 255.0];
        let features =
            Tensor::from_vec(features.to_vec(), (1, 4), &self.device)?.to_dtype(self.dtype)?;
        let pooled = hidden.i((.., 0))?;
        let action = Tensor::cat(&[pooled, features], 1)?;
        let action = self
            .action_out
            .forward(&self.action_in.forward(&action)?.gelu_erf()?)?
            .to_dtype(DType::F32)?;
        let action = candle_nn::ops::softmax_last_dim(&action)?.to_vec2::<f32>()?[0][0];
        Ok(LayaOutput {
            logits,
            action_probability: action,
        })
    }
}

impl super::Model for LayaModel {
    fn decide(
        &self,
        mut batch: Batch,
        inputs: Vec<infinity_v3_backend_core::DecisionInput>,
    ) -> Result<Vec<infinity_v3_backend_core::DecisionOutput>> {
        if inputs.len() != batch.len() {
            candle::bail!("Laya metadata count does not match batch")
        }
        let inputs = inputs
            .into_iter()
            .map(|input| match input {
                infinity_v3_backend_core::DecisionInput::Laya {
                    question_type,
                    markers,
                } => Ok((question_type, markers)),
                infinity_v3_backend_core::DecisionInput::Warmup => Ok((0, vec![0])),
                _ => candle::bail!("Laya requires marker decision metadata"),
            })
            .collect::<Result<Vec<_>>>()?;
        let lengths = batch.cumulative_seq_lengths.clone();
        for (i, input) in inputs.iter().enumerate() {
            let length = (lengths[i + 1] - lengths[i]) as usize;
            if input.0 > 2 || input.1.is_empty() || input.1.iter().any(|&p| p >= length) {
                candle::bail!("invalid Laya decision input")
            }
        }
        batch.pooled_indices.clear();
        batch.raw_indices = (0..inputs.len() as u32).collect();
        // Run ModernBERT once for the entire queue batch. Heads use unpadded
        // per-question states so options never attend to another question.
        let (_, raw) = self.encoder.forward(batch)?;
        let raw =
            raw.ok_or_else(|| candle::Error::Msg("Laya encoder returned no hidden states".into()))?;
        inputs
            .iter()
            .enumerate()
            .map(|(i, input)| {
                let length = (lengths[i + 1] - lengths[i]) as usize;
                let hidden = raw.narrow(0, lengths[i] as usize, length)?.reshape((
                    1,
                    length,
                    self.hidden,
                ))?;
                let output = self.forward_head(hidden, &input.1, input.0)?;
                Ok(infinity_v3_backend_core::DecisionOutput {
                    memory: None,
                    logits: output.logits,
                    action_probability: output.action_probability,
                })
            })
            .collect()
    }
}
