//! Vision feature injection for Rune's packed Gemma4 decision path.
use candle::{Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma4::{
    config::Gemma4VisionConfig, multimodal_embedding::MultimodalEmbedder, vision::VisionTower,
};
use infinity_v3_backend_core::Batch;
use sha2::{Digest, Sha256};
use std::{collections::VecDeque, sync::Mutex};

const FEATURE_CACHE_BYTES: usize = 64 * 1024 * 1024;
const FEATURE_CACHE_ENTRIES: usize = 32;

#[derive(Default)]
struct FeatureCache {
    entries: VecDeque<([u8; 32], Tensor)>,
    bytes: usize,
}

impl FeatureCache {
    fn get(&mut self, key: &[u8; 32]) -> Option<Tensor> {
        let index = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(index)?;
        let value = entry.1.clone();
        self.entries.push_back(entry);
        Some(value)
    }

    fn insert(&mut self, key: [u8; 32], value: Tensor) {
        let size = value.elem_count() * value.dtype().size_in_bytes();
        if size > FEATURE_CACHE_BYTES {
            return;
        }
        if let Some(index) = self.entries.iter().position(|(k, _)| k == &key) {
            let (_, old) = self.entries.remove(index).unwrap();
            self.bytes -= old.elem_count() * old.dtype().size_in_bytes();
        }
        while self.bytes + size > FEATURE_CACHE_BYTES || self.entries.len() >= FEATURE_CACHE_ENTRIES
        {
            let (_, old) = self.entries.pop_front().unwrap();
            self.bytes -= old.elem_count() * old.dtype().size_in_bytes();
        }
        self.bytes += size;
        self.entries.push_back((key, value));
    }
}

fn image_key(image: &infinity_v3_backend_core::ImagePatches) -> [u8; 32] {
    let mut hash = Sha256::new();
    for value in image
        .grid_thw
        .into_iter()
        .chain([image.patch_dim, image.merge_size])
    {
        hash.update(value.to_le_bytes());
    }
    // Hash the actual prepared pixels and layout; URLs and image placeholders
    // cannot identify model inputs. The cache belongs to one model/device.
    hash.update(bytemuck::cast_slice(&image.pixels));
    hash.finalize().into()
}

pub(super) struct Gemma4Vision {
    tower: VisionTower,
    projection: MultimodalEmbedder,
    features: Mutex<FeatureCache>,
}

impl Gemma4Vision {
    pub(super) fn load(
        vb: VarBuilder,
        config: &serde_json::Value,
        hidden_size: usize,
    ) -> Result<Self> {
        Self::load_at(vb, config, hidden_size, "model")
    }

    pub(super) fn load_at(
        vb: VarBuilder,
        config: &serde_json::Value,
        hidden_size: usize,
        prefix: &str,
    ) -> Result<Self> {
        if config["use_clipped_linears"].as_bool().unwrap_or(false) {
            candle::bail!("Clipped Gemma4 vision projections are unsupported");
        }
        let config: Gemma4VisionConfig =
            serde_json::from_value(config.clone()).map_err(candle::Error::wrap)?;
        if config.patch_size != 16 || config.pooling_kernel_size != 3 {
            candle::bail!("Unsupported Gemma4 vision patch layout");
        }
        let prefix = if prefix.is_empty() {
            String::new()
        } else {
            format!("{prefix}.")
        };
        // Transformers wraps even unclipped vision projections in `linear`.
        let encoder_prefix = format!("{prefix}vision_tower.encoder.");
        let vision_vb = vb.clone().rename_f(move |name| {
            if name.starts_with(&encoder_prefix) && name.ends_with("_proj.weight") {
                format!("{}.linear.weight", name.trim_end_matches(".weight"))
            } else {
                name.to_owned()
            }
        });
        Ok(Self {
            features: Mutex::new(FeatureCache::default()),
            tower: VisionTower::new(&config, vision_vb.pp(format!("{prefix}vision_tower")))?,
            projection: MultimodalEmbedder::new(
                config.hidden_size,
                hidden_size,
                config.rms_norm_eps,
                vb.pp(format!("{prefix}embed_vision")),
            )?,
        })
    }

    pub(super) fn inject(
        &self,
        batch: &Batch,
        embeddings: &mut Tensor,
    ) -> Result<Vec<(usize, usize, usize)>> {
        if batch.multimodal.len() != batch.len() {
            candle::bail!("Invalid Gemma4 image batch metadata");
        }
        let mut spans = Vec::new();
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
                    || h % 3 != 0
                    || w % 3 != 0
                    || image.merge_size != 3
                    || image.patch_dim != 768
                    || image.pixels.len() != h * w * 768
                    || *start < previous_end
                    || start + image.token_count() > length
                {
                    candle::bail!("Invalid Gemma4 image patches");
                }
                let features = if let Some(features) =
                    features_by_image.get(&std::sync::Arc::as_ptr(image))
                {
                    Tensor::clone(features)
                } else {
                    let key = image_key(image);
                    let cached = self
                        .features
                        .lock()
                        .map_err(|_| candle::Error::Msg("Image feature cache poisoned".into()))?
                        .get(&key);
                    let features = if let Some(features) = cached {
                        features
                    } else {
                        let pv = Tensor::from_slice(
                            &image.pixels,
                            (1, h, w, 16, 16, 3),
                            embeddings.device(),
                        )?
                        .permute((0, 5, 1, 3, 2, 4))?
                        .contiguous()?
                        .reshape((1, 3, h * 16, w * 16))?;
                        let features = self
                            .projection
                            .forward(&self.tower.forward(&[pv])?)?
                            .squeeze(0)?;
                        if features.dim(0)? != image.token_count() {
                            candle::bail!("Gemma4 vision token count mismatch");
                        }
                        self.features
                            .lock()
                            .map_err(|_| candle::Error::Msg("Image feature cache poisoned".into()))?
                            .insert(key, features.clone());
                        features
                    };
                    features_by_image.insert(std::sync::Arc::as_ptr(image), features.clone());
                    features
                };
                *embeddings = embeddings.slice_scatter0(&features, offset + start)?;
                spans.push((offset + start, image.token_count(), offset));
                previous_end = start + image.token_count();
            }
        }
        Ok(spans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use std::{collections::BTreeSet, path::PathBuf};

    #[test]
    fn feature_cache_tracks_content_and_evicts_old_entries() -> Result<()> {
        let image = infinity_v3_backend_core::ImagePatches {
            pixels: vec![0.; 12],
            grid_thw: [1, 2, 2],
            patch_dim: 3,
            merge_size: 1,
        };
        assert_eq!(image_key(&image), image_key(&image.clone()));
        let mut changed = image.clone();
        changed.pixels[11] = 1.;
        assert_ne!(image_key(&image), image_key(&changed));
        changed = image.clone();
        changed.grid_thw = [1, 1, 4];
        assert_ne!(image_key(&image), image_key(&changed));
        let mut cache = FeatureCache::default();
        let value = Tensor::zeros((2, 4), DType::F32, &Device::Cpu)?;
        for id in 0..FEATURE_CACHE_ENTRIES {
            cache.insert([id as u8; 32], value.clone());
        }
        assert!(cache.get(&[0; 32]).is_some());
        cache.insert([255; 32], value.clone());
        assert!(cache.get(&[1; 32]).is_none());
        assert!(cache.get(&[0; 32]).is_some());
        cache.insert([255; 32], value);
        assert_eq!(cache.entries.len(), FEATURE_CACHE_ENTRIES);
        assert_eq!(cache.bytes, FEATURE_CACHE_ENTRIES * 2 * 4 * 4);
        let mut cache = FeatureCache::default();
        let large = Tensor::zeros(FEATURE_CACHE_BYTES / 8 + 1, DType::F32, &Device::Cpu)?;
        cache.insert([0; 32], large.clone());
        cache.insert([1; 32], large);
        assert!(cache.get(&[0; 32]).is_none());
        assert!(cache.get(&[1; 32]).is_some());
        assert!(cache.bytes <= FEATURE_CACHE_BYTES);
        let oversized = Tensor::zeros(FEATURE_CACHE_BYTES / 4 + 1, DType::F32, &Device::Cpu)?;
        cache.insert([2; 32], oversized);
        assert!(cache.get(&[2; 32]).is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires Rune checkpoint, Transformers pixel/features fixtures, and CUDA"]
    fn rune_vision_matches_reference() -> anyhow::Result<()> {
        let root = PathBuf::from(std::env::var("RUNE_CHECKPOINT_DIR")?);
        let fixture = PathBuf::from(std::env::var("RUNE_IMAGE_FIXTURE_DIR")?);
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json"))?)?;
        let paths: BTreeSet<_> = index["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| root.join(v.as_str().unwrap()))
            .collect();
        let device = Device::new_cuda(0)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &paths.into_iter().collect::<Vec<_>>(),
                DType::BF16,
                &device,
            )?
        };
        let vision = Gemma4Vision::load(
            vb,
            &config["vision_config"],
            config["text_config"]["hidden_size"].as_u64().unwrap() as usize,
        )?;
        for name in ["red", "green", "blue"] {
            let input = candle::safetensors::load(
                fixture.join(format!("{name}.inputs.safetensors")),
                &device,
            )?;
            let positions = input["image_position_ids"]
                .to_dtype(DType::I64)?
                .flatten_all()?
                .to_vec1::<i64>()?;
            let w = positions.chunks_exact(2).map(|p| p[0]).max().unwrap() as usize + 1;
            let h = positions.chunks_exact(2).map(|p| p[1]).max().unwrap() as usize + 1;
            let pixels = input["pixel_values"]
                .narrow(1, 0, h * w)?
                .reshape((1, h, w, 16, 16, 3))?
                .permute((0, 5, 1, 3, 2, 4))?
                .contiguous()?
                .reshape((1, 3, h * 16, w * 16))?;
            let actual = vision
                .projection
                .forward(&vision.tower.forward(&[pixels])?)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            let expected = candle::safetensors::load(
                fixture.join(format!("{name}.outputs.safetensors")),
                &device,
            )?
            .remove("features")
            .unwrap();
            let a = actual.flatten_all()?.to_vec1::<f32>()?;
            let b = expected.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(a.len(), b.len());
            let dot: f64 = a
                .iter()
                .zip(&b)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let norm = |x: &[f32]| x.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (norm(&a) * norm(&b));
            println!("{name}: vision feature cosine {cosine}");
            assert!(cosine > 0.995, "{name}: {cosine}");
        }
        Ok(())
    }
}
