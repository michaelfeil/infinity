//! Packed Qwen3-VL image embeddings using the existing Qwen3 text layers.
use crate::models::{FlashQwen3Model, Model, Qwen3Config};
use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_vl::{config::VisionConfig, vision::Qwen3VLVisionModel};
use infinity_v3_backend_core::{Batch, ModelType, Pool};

pub struct Qwen3VlModel {
    text: FlashQwen3Model,
    vision: Qwen3VLVisionModel,
    device: Device,
    dtype: DType,
    head_dim: usize,
    inv_freq: Vec<f32>,
    sections: [usize; 3],
    merge_size: usize,
}

impl Qwen3VlModel {
    pub fn load(vb: VarBuilder, config: serde_json::Value, model_type: ModelType) -> Result<Self> {
        if vb.dtype() != DType::F16 {
            candle::bail!("Qwen3-VL currently requires float16; bfloat16 parity is not validated");
        }
        if model_type != ModelType::Embedding(Pool::LastToken) {
            candle::bail!("Qwen3-VL embeddings require last-token pooling");
        }
        let mut text = config["text_config"].clone();
        text["use_sliding_window"] = false.into();
        let sections: [usize; 3] =
            serde_json::from_value(text["rope_scaling"]["mrope_section"].clone())
                .map_err(candle::Error::wrap)?;
        if text["rope_scaling"]["mrope_interleaved"] != true {
            candle::bail!("Only Qwen3-VL interleaved multimodal RoPE is supported");
        }
        let text: Qwen3Config = serde_json::from_value(text).map_err(candle::Error::wrap)?;
        let head_dim = text
            .head_dim
            .unwrap_or(text.hidden_size / text.num_attention_heads);
        if sections.iter().sum::<usize>() != head_dim / 2 {
            candle::bail!("Invalid multimodal RoPE sections");
        }
        let vision: VisionConfig =
            serde_json::from_value(config["vision_config"].clone()).map_err(candle::Error::wrap)?;
        Ok(Self {
            text: FlashQwen3Model::load(vb.pp("model.language_model"), &text, model_type, false)?,
            vision: Qwen3VLVisionModel::new(&vision, vb.pp("model.visual"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            head_dim,
            inv_freq: (0..head_dim / 2)
                .map(|j| {
                    (1.0 / (text.rope_theta as f64).powf((2 * j) as f64 / head_dim as f64)) as f32
                })
                .collect(),
            sections,
            merge_size: vision.spatial_merge_size,
        })
    }
}

impl Model for Qwen3VlModel {
    // Text token identity is insufficient to establish equivalent visual prefixes.
    fn supports_radix_mlp(&self) -> bool {
        false
    }
    fn predict(&self, _batch: Batch) -> Result<Tensor> {
        candle::bail!("Qwen3-VL embedding checkpoint cannot classify")
    }
    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if !batch.raw_indices.is_empty() {
            candle::bail!("Qwen3-VL exposes pooled embeddings only");
        }
        if batch.compact_input_ids.is_some() {
            candle::bail!("Image inputs cannot use token-only Radix folding");
        }
        if !batch.multimodal.is_empty() && batch.multimodal.len() != batch.len() {
            candle::bail!("Missing multimodal preprocessing metadata");
        }
        let pixel_count = batch
            .multimodal
            .iter()
            .flatten()
            .flat_map(|media| &media.images)
            .map(|(_, image)| image.pixels.len())
            .sum();
        let mut pixels = Vec::with_capacity(pixel_count);
        let mut grids = Vec::new();
        let mut indices = Vec::new();
        let mut positions = [Vec::new(), Vec::new(), Vec::new()];
        let mut patch_dim = None;
        for i in 0..batch.len() {
            let offset = batch.cumulative_seq_lengths[i] as usize;
            let length =
                (batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i]) as usize;
            // Backend warmup/health probes contain ordinary token IDs without media.
            let Some(media) = batch.multimodal.get(i).and_then(Option::as_ref) else {
                for axis in &mut positions {
                    axis.extend_from_slice(&batch.position_ids[offset..offset + length]);
                }
                continue;
            };
            for (axis, positions) in positions.iter_mut().enumerate() {
                if media.position_ids[axis].len() != length {
                    candle::bail!("Invalid image position length");
                }
                positions.extend_from_slice(&media.position_ids[axis]);
            }
            for (start, image) in &media.images {
                if image.merge_size != self.merge_size
                    || patch_dim.is_some_and(|d| d != image.patch_dim)
                {
                    candle::bail!("Image processor does not match vision model");
                }
                patch_dim = Some(image.patch_dim);
                if image.pixels.len() != image.grid_thw.iter().product::<usize>() * image.patch_dim
                    || start + image.token_count() > length
                {
                    candle::bail!("Invalid image patches");
                }
                pixels.extend_from_slice(&image.pixels);
                grids.extend(image.grid_thw.map(|v| v as u32));
                indices.extend(
                    (offset + start..offset + start + image.token_count()).map(|v| v as u32),
                );
            }
        }
        let half = self.head_dim / 2;
        let mut cos = Vec::with_capacity(batch.input_ids.len() * half);
        let mut sin = Vec::with_capacity(cos.capacity());
        for ((temporal, height), width) in positions[0]
            .iter()
            .zip(&positions[1])
            .zip(&positions[2])
            .take(batch.input_ids.len())
        {
            let position = [*temporal, *height, *width];
            for (j, frequency) in self.inv_freq.iter().enumerate() {
                let axis = if j % 3 == 1 && j < self.sections[1] * 3 {
                    1
                } else if j % 3 == 2 && j < self.sections[2] * 3 {
                    2
                } else {
                    0
                };
                let angle = position[axis] as f32 * frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let cos = Tensor::from_vec(cos, (batch.input_ids.len(), half), &self.device)?
            .to_dtype(self.dtype)?;
        let sin = Tensor::from_vec(sin, (batch.input_ids.len(), half), &self.device)?
            .to_dtype(self.dtype)?;
        let visual = if let Some(dim) = patch_dim {
            let rows = pixels.len() / dim;
            let pixels =
                Tensor::from_vec(pixels, (rows, dim), &self.device)?.to_dtype(self.dtype)?;
            let images = grids.len() / 3;
            let grid = Tensor::from_vec(grids, (images, 3), &self.device)?;
            let (features, deepstack) = self.vision.forward(&pixels, &grid)?;
            Some((
                Tensor::new(indices.as_slice(), &self.device)?,
                features,
                deepstack,
            ))
        } else {
            None
        };
        let pooled = self.text.multimodal_embeddings(
            &batch,
            visual.as_ref().map(|(i, f, d)| (i, f, d.as_slice())),
            &cos,
            &sin,
        )?;
        Ok((Some(pooled), None))
    }
}
