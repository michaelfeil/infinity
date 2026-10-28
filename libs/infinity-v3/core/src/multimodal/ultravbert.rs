//! UltraVBERT's fixed SigLIP tiles and Idefics image token layout.
use super::{
    image::QwenImageProcessor,
    media::{MediaBudget, MediaResolver},
    MultimodalConfig,
};
use crate::TextEmbeddingsError;
use infinity_v3_backend::ImagePatches;
use std::{path::Path, sync::Arc};
use tokenizers::Tokenizer;

fn invalid(s: impl Into<String>) -> TextEmbeddingsError {
    TextEmbeddingsError::Validation(s.into())
}

#[derive(Clone)]
pub struct UltraVbertProcessor {
    tokenizer: Tokenizer,
    resolver: MediaResolver,
    budget: MediaBudget,
    config: MultimodalConfig,
    workers: Arc<tokio::sync::Semaphore>,
    image_id: u32,
}
pub struct PreparedUltraVbert {
    pub prefix: Vec<u32>,
    pub images: Vec<(usize, Arc<ImagePatches>)>,
    pub memory: Arc<tokio::sync::OwnedSemaphorePermit>,
}
impl UltraVbertProcessor {
    pub fn load(
        root: &Path,
        tokenizer: Tokenizer,
        config: MultimodalConfig,
    ) -> Result<Self, TextEmbeddingsError> {
        let metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("processor_config.json"))
                .map_err(|e| invalid(e.to_string()))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let image = &metadata["image_processor"];
        if !(metadata["image_seq_len"] == 64
            && image["resample"] == 1
            && image["max_image_size"]["longest_edge"] == 512
            && image["image_mean"] == serde_json::json!([0.5, 0.5, 0.5])
            && image["image_std"] == serde_json::json!([0.5, 0.5, 0.5])
            && image["do_resize"] == true
            && image["do_image_splitting"] == true
            && image["do_rescale"] == true
            && image["do_normalize"] == true)
        {
            return Err(invalid("Unsupported UltraVBERT image processor"));
        }
        if config.workers == 0 || config.workers > 32 {
            return Err(invalid("Invalid image worker count"));
        }
        let budget = MediaBudget::new(config.memory_budget_bytes)?;
        let resolver = MediaResolver::new(
            config.allowed_image_hosts.clone(),
            budget.clone(),
            config.max_image_bytes,
            config.download_concurrency,
            config.timeout,
        )?;
        let image_id = tokenizer
            .token_to_id("<image>")
            .ok_or_else(|| invalid("Missing UltraVBERT image token"))?;
        Ok(Self {
            tokenizer,
            resolver,
            budget,
            workers: Arc::new(tokio::sync::Semaphore::new(config.workers)),
            config,
            image_id,
        })
    }

    pub async fn prepare(
        &self,
        sources: Vec<String>,
    ) -> Result<PreparedUltraVbert, TextEmbeddingsError> {
        if sources.is_empty() || sources.len() > 3 || sources.len() > self.config.max_images {
            return Err(invalid("UltraVBERT reads 1-3 images per request"));
        }
        let mut images = Vec::new();
        for source in sources {
            images.push(self.resolver.resolve(source).await?);
        }
        let worker = self
            .workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| invalid("Image workers closed"))?;
        // Inspect dimensions before decoding, and admit decoder scratch as well
        // as resized tiles, packed F32 pixels and the upload copy.
        let bytes = images.iter().try_fold(0usize, |total, image| {
            use image::ImageDecoder;
            let decoder =
                QwenImageProcessor::decoder(&image.bytes, self.config.max_decoded_pixels as u64)
                    .map_err(invalid)?;
            let (w, h) = decoder.dimensions();
            (w as usize)
                .checked_mul(h as usize)
                .and_then(|n| n.checked_mul(16))
                // Horizontal-first Lanczos resizing retains target_width ×
                // source_height RGB rows before its vertical pass.
                .and_then(|n| n.checked_add(1024 * h as usize * 3))
                .and_then(|n| n.checked_add(5 * 512 * 512 * 3 * 12))
                .and_then(|n| total.checked_add(n))
                .ok_or_else(|| invalid("Image memory estimate overflow"))
        })?;
        let memory = Arc::new(self.budget.reserve(bytes)?);
        let this = self.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::task::spawn_blocking(move || {
            let _worker = worker;
            let result = (|| {
                let mut prompt = String::new();
                let mut tiles = Vec::new();
                for image in images {
                    if tx.is_closed() {
                        return Err(invalid("Image request canceled"));
                    }
                    let decoder = QwenImageProcessor::decoder(
                        &image.bytes,
                        this.config.max_decoded_pixels as u64,
                    )
                    .map_err(invalid)?;
                    use image::ImageDecoder;
                    let (width, height) = decoder.dimensions();
                    let decoded = image::DynamicImage::from_decoder(decoder)
                        .map_err(|_| invalid("Invalid image"))?
                        .into_rgb8();
                    let ratio = f64::from(width) / f64::from(height);
                    let (w, h) = if width >= height {
                        (1024, even((1024. / ratio) as u32))
                    } else {
                        (even((1024. * ratio) as u32), 1024)
                    };
                    let resized = resize(decoded, w, h)?;
                    let cols = w.div_ceil(512);
                    let rows = h.div_ceil(512);
                    let full = resize(resized, cols * 512, rows * 512)?;
                    for row in 0..rows {
                        for col in 0..cols {
                            prompt.push_str(&format!(
                                "<fake_token_around_image><row_{}_col_{}>{}",
                                row + 1,
                                col + 1,
                                "<image>".repeat(64)
                            ));
                            tiles.push(
                                image::imageops::crop_imm(&full, col * 512, row * 512, 512, 512)
                                    .to_image(),
                            );
                        }
                        prompt.push('\n');
                    }
                    prompt.push_str(&format!(
                        "\n<fake_token_around_image><global-img>{}<fake_token_around_image>",
                        "<image>".repeat(64)
                    ));
                    tiles.push(resize(full, 512, 512)?);
                }
                let prefix = this
                    .tokenizer
                    .encode(prompt, false)
                    .map_err(|e| invalid(e.to_string()))?
                    .get_ids()
                    .to_vec();
                let mut prefix_with_bos = vec![this
                    .tokenizer
                    .token_to_id("<|endoftext|>")
                    .ok_or_else(|| invalid("Missing BOS token"))?];
                prefix_with_bos.extend(prefix);
                let starts: Vec<_> = prefix_with_bos
                    .iter()
                    .enumerate()
                    .filter_map(|(i, id)| {
                        if *id == this.image_id
                            && (i == 0 || prefix_with_bos.get(i - 1) != Some(&this.image_id))
                        {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();
                if starts.len() != tiles.len() {
                    return Err(invalid("UltraVBERT image placeholder mismatch"));
                }
                let mut patches = Vec::new();
                for (start, tile) in starts.into_iter().zip(tiles) {
                    if tx.is_closed() {
                        return Err(invalid("Image request canceled"));
                    }
                    let mut pixels = Vec::with_capacity(512 * 512 * 3);
                    for row in 0..32 {
                        for col in 0..32 {
                            for c in 0..3 {
                                for y in 0..16 {
                                    for x in 0..16 {
                                        pixels.push(
                                            (f32::from(
                                                tile.get_pixel(col * 16 + x, row * 16 + y)[c],
                                            ) / 255.
                                                - 0.5)
                                                / 0.5,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    patches.push((
                        start,
                        Arc::new(ImagePatches {
                            pixels,
                            grid_thw: [1, 32, 32],
                            patch_dim: 768,
                            merge_size: 4,
                        }),
                    ));
                }
                Ok(PreparedUltraVbert {
                    prefix: prefix_with_bos,
                    images: patches,
                    memory,
                })
            })();
            let _ = tx.send(result);
        });
        rx.await.map_err(|_| invalid("Image worker failed"))?
    }
}

fn even(n: u32) -> u32 {
    n.max(1).saturating_add(n.max(1) % 2)
}

fn resize(
    rgb: image::RgbImage,
    width: u32,
    height: u32,
) -> Result<image::RgbImage, TextEmbeddingsError> {
    let source = fast_image_resize::images::Image::from_vec_u8(
        rgb.width(),
        rgb.height(),
        rgb.into_raw(),
        fast_image_resize::PixelType::U8x3,
    )
    .map_err(|_| invalid("Invalid RGB image"))?;
    let mut intermediate = fast_image_resize::images::Image::new(
        width,
        source.height(),
        fast_image_resize::PixelType::U8x3,
    );
    let mut output =
        fast_image_resize::images::Image::new(width, height, fast_image_resize::PixelType::U8x3);
    let options = fast_image_resize::ResizeOptions::new().resize_alg(
        fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Lanczos3),
    );
    let mut resizer = fast_image_resize::Resizer::new();
    resizer
        .resize(&source, &mut intermediate, &options)
        .map_err(|_| invalid("Image resize failed"))?;
    resizer
        .resize(&intermediate, &mut output, &options)
        .map_err(|_| invalid("Image resize failed"))?;
    image::RgbImage::from_raw(width, height, output.into_vec())
        .ok_or_else(|| invalid("Invalid resized image"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    #[tokio::test]
    #[ignore = "requires upstream-generated checkpoint fixtures in VELA3_REFERENCE_DIR"]
    async fn checkpoint_image_layout_and_pixels_match_reference(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = std::path::PathBuf::from(std::env::var("VELA3_REFERENCE_DIR")?);
        let path = root.join("vela3-fast-vision");
        let tok = Tokenizer::from_file(path.join("tokenizer.json")).map_err(|e| e.to_string())?;
        let processor = UltraVbertProcessor::load(&path, tok, Default::default())?;
        let refs: serde_json::Value = serde_json::from_slice(&std::fs::read(
            root.join("vela3-fast-vision-reference.json"),
        )?)?;
        for (case, files) in [
            ("image", vec!["square.png"]),
            ("wide", vec!["wide.png"]),
            ("two_images", vec!["square.png", "wide.png"]),
        ] {
            let sources = files
                .into_iter()
                .map(|name| {
                    format!(
                        "data:image/png;base64,{}",
                        base64::engine::general_purpose::STANDARD
                            .encode(std::fs::read(root.join(name)).unwrap())
                    )
                })
                .collect();
            let prepared = processor.prepare(sources).await?;
            let ids: Vec<u32> =
                serde_json::from_value(refs[case]["records"]["route"]["ids"].clone())?;
            assert_eq!(
                prepared.prefix,
                ids[..prepared.prefix.len()],
                "{case} image prefix"
            );
            let data = std::fs::read(root.join(format!("vela3-fast-vision-{case}-pixels.pt.bin")))?;
            let expected: Vec<f32> = data
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            let pixels: Vec<f32> = prepared
                .images
                .iter()
                .flat_map(|(_, image)| image.pixels.iter().copied())
                .collect();
            assert_eq!(pixels.len(), expected.len(), "{case} pixels");
            let error = pixels
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                error <= 2. / 255. + 1e-6,
                "{case} max normalized pixel error {error}"
            );
        }
        // A narrow portrait's horizontal resize scratch exceeds 64 MiB even
        // though its decoded pixel count is tiny. Reject it before decoding.
        let limited = UltraVbertProcessor::load(
            &path,
            processor.tokenizer.clone(),
            MultimodalConfig {
                memory_budget_bytes: 64 << 20,
                ..Default::default()
            },
        )?;
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbImage::new(1, 16384).write_to(&mut png, image::ImageFormat::Png)?;
        let source = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png.into_inner())
        );
        assert!(
            matches!(limited.prepare(vec![source]).await,Err(TextEmbeddingsError::Validation(message)) if message.contains("memory budget"))
        );
        Ok(())
    }
}
