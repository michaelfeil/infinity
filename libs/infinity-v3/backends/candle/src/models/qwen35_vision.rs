//! Qwen3.5 uses the Qwen3-VL vision tower without DeepStack features.
use super::qwen35_config::Qwen35TextConfig;
use candle::{DType, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_vl::{config::VisionConfig, vision::Qwen3VLVisionModel};
use infinity_v3_backend_core::Batch;

pub(super) struct Qwen35Vision {
    tower: Qwen3VLVisionModel,
    dtype: DType,
    sections: [usize; 3],
    patch_dim: usize,
    merge_size: usize,
}

impl Qwen35Vision {
    pub(super) fn load(
        vb: VarBuilder,
        value: &serde_json::Value,
        text: &Qwen35TextConfig,
    ) -> Result<Self> {
        let config: VisionConfig =
            serde_json::from_value(value.clone()).map_err(candle::Error::wrap)?;
        let sections = text
            .rope_parameters
            .mrope_section
            .ok_or_else(|| candle::Error::Msg("Missing multimodal RoPE sections".into()))?;
        if !config.deepstack_visual_indexes.is_empty()
            || config.num_heads == 0
            || config.num_heads > 16
            || !config.hidden_size.is_multiple_of(config.num_heads)
            || config.patch_size != 16
            || config.temporal_patch_size != 2
            || config.spatial_merge_size != 2
            || config.in_chans != 3
            || value["in_channels"].as_u64().is_some_and(|n| n != 3)
            || config.out_hidden_size != text.hidden_size
            || !text.rope_parameters.mrope_interleaved
            || sections.iter().sum::<usize>() != text.rotary_dim() / 2
        {
            candle::bail!("Unsupported Qwen3.5 vision configuration");
        }
        let vision_dtype = if vb.dtype() == DType::BF16 {
            DType::F16
        } else {
            vb.dtype()
        };
        let vision = if vb.contains_tensor("model.visual.patch_embed.proj.weight") {
            vb.pp("model.visual")
        } else if vb.contains_tensor("visual.patch_embed.proj.weight") {
            vb.pp("visual")
        } else {
            vb.pp("model.visual")
        };
        Ok(Self {
            // FP16 retains more mantissa precision in the vision tower than BF16
            // while keeping its projections on 16-bit tensor cores.
            tower: Qwen3VLVisionModel::new(&config, vision.to_dtype(vision_dtype))?,
            dtype: vision_dtype,
            sections,
            patch_dim: config.in_chans
                * config.temporal_patch_size
                * config.patch_size
                * config.patch_size,
            merge_size: config.spatial_merge_size,
        })
    }

    pub(super) fn inject(&self, batch: &Batch, embeddings: &mut Tensor) -> Result<()> {
        if batch.multimodal.len() != batch.len() {
            candle::bail!("Invalid Qwen3.5 image batch metadata");
        }
        let mut features_by_image = std::collections::HashMap::new();
        for (index, media) in batch.multimodal.iter().enumerate() {
            let Some(media) = media else { continue };
            let offset = batch.cumulative_seq_lengths[index] as usize;
            let length = (batch.cumulative_seq_lengths[index + 1]
                - batch.cumulative_seq_lengths[index]) as usize;
            let mut previous_end = 0;
            for (start, image) in &media.images {
                let [t, h, w] = image.grid_thw;
                if t != 1
                    || h == 0
                    || w == 0
                    || h % self.merge_size != 0
                    || w % self.merge_size != 0
                    // The reused tower materializes FP32 attention scores. Keep
                    // its per-image attention allocation below 1 GiB at 16 heads.
                    || h * w > 4096
                    || image.merge_size != self.merge_size
                    || image.patch_dim != self.patch_dim
                    || image.pixels.len() != h * w * self.patch_dim
                    || *start < previous_end
                    || start + image.token_count() > length
                {
                    candle::bail!("Invalid Qwen3.5 image patches");
                }
                let features = if let Some(features) =
                    features_by_image.get(&std::sync::Arc::as_ptr(image))
                {
                    Tensor::clone(features)
                } else {
                    let pixels = Tensor::from_slice(
                        &image.pixels,
                        (h * w, self.patch_dim),
                        embeddings.device(),
                    )?
                    .to_dtype(self.dtype)?;
                    let grid = Tensor::new(&[[t as u32, h as u32, w as u32]], embeddings.device())?;
                    let (features, _) = self.tower.forward(&pixels, &grid)?;
                    let features = features.to_dtype(embeddings.dtype())?;
                    if features.dim(0)? != image.token_count()
                        || features.dim(1)? != embeddings.dim(1)?
                    {
                        candle::bail!("Qwen3.5 vision feature shape mismatch");
                    }
                    features_by_image.insert(std::sync::Arc::as_ptr(image), features.clone());
                    features
                };
                *embeddings = embeddings.slice_scatter0(&features, offset + start)?;
                previous_end = start + image.token_count();
            }
        }
        Ok(())
    }

    pub(super) fn rope(
        &self,
        batch: &Batch,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let width = cos.dim(1)?;
        let half = width / 2;
        let mut indices = Vec::with_capacity(batch.input_ids.len() * width);
        for (index, media) in batch.multimodal.iter().enumerate() {
            let start = batch.cumulative_seq_lengths[index] as usize;
            let end = batch.cumulative_seq_lengths[index + 1] as usize;
            if media
                .as_ref()
                .is_some_and(|m| m.position_ids.iter().any(|axis| axis.len() != end - start))
            {
                candle::bail!("Invalid Qwen3.5 image positions");
            }
            for token in start..end {
                for column in 0..width {
                    let frequency = column % half;
                    let axis = if frequency % 3 == 1 && frequency < self.sections[1] * 3 {
                        1
                    } else if frequency % 3 == 2 && frequency < self.sections[2] * 3 {
                        2
                    } else {
                        0
                    };
                    let position = match media {
                        Some(media) => media.position_ids[axis][token - start],
                        None => batch.position_ids[token],
                    } as usize;
                    if position >= cos.dim(0)? {
                        candle::bail!("Qwen3.5 image position exceeds model context");
                    }
                    indices.push((position * width + column) as u32);
                }
            }
        }
        let indices = Tensor::new(indices.as_slice(), cos.device())?;
        let gather = |cache: &Tensor| {
            cache
                .flatten_all()?
                .index_select(&indices, 0)?
                .reshape((batch.input_ids.len(), width))
        };
        Ok((gather(cos)?, gather(sin)?))
    }
}
