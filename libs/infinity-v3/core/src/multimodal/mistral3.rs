//! Public Mistral3/Pixtral image preparation, matching Transformers 5.12.1.
//! Adapted from Transformers' Apache-2.0 Pixtral processor; see retained notices.
use super::{
    image::{resize_rgb, QwenImageProcessor},
    media::{MediaBudget, MediaResolver, ResolvedImage},
    MultimodalConfig, PreparedMultimodal,
};
use crate::{
    chat::ChatProcessor,
    input::{ContentPart, ImageDetail, Message, MessageContent, MessageRole},
    tokenization::EncodingInput,
    TextEmbeddingsError,
};
use image::ImageDecoder;
use infinity_v3_backend::{ImagePatches, MultimodalEncoding};
use std::{path::Path, sync::Arc};
use tokenizers::{Tokenizer, TruncationDirection};
use tokio::sync::Semaphore;

type Result<T> = std::result::Result<T, TextEmbeddingsError>;
fn invalid(s: impl Into<String>) -> TextEmbeddingsError {
    TextEmbeddingsError::Validation(s.into())
}

#[derive(Clone)]
pub struct Mistral3Processor {
    state: Arc<State>,
    resolver: MediaResolver,
    workers: Arc<Semaphore>,
}
impl std::fmt::Debug for Mistral3Processor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mistral3Processor").finish_non_exhaustive()
    }
}
struct State {
    chat: ChatProcessor,
    tokenizer: Tokenizer,
    config: MultimodalConfig,
    budget: MediaBudget,
    max_tokens: usize,
    patch: usize,
    merge: usize,
    edge: u32,
    normalized: [[f32; 256]; 3],
    ids: [u32; 3],
}
impl Mistral3Processor {
    pub fn load(
        root: &Path,
        mut tokenizer: Tokenizer,
        max_tokens: usize,
        config: MultimodalConfig,
    ) -> Result<Self> {
        let read = |name: &str| -> Result<serde_json::Value> {
            serde_json::from_slice(
                &std::fs::read(root.join(name)).map_err(|e| invalid(e.to_string()))?,
            )
            .map_err(|e| invalid(e.to_string()))
        };
        let model = read("config.json")?;
        let processor = read("processor_config.json")?;
        let image = &processor["image_processor"];
        let patch = model["vision_config"]["patch_size"].as_u64().unwrap_or(0) as usize;
        let merge = model["spatial_merge_size"].as_u64().unwrap_or(2) as usize;
        let edge = image["size"]["longest_edge"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0);
        if model["model_type"] != "mistral3"
            || patch == 0
            || patch > 64
            || merge == 0
            || merge > 8
            || edge == 0
            || edge > 4096
            || max_tokens == 0
            || max_tokens > u32::MAX as usize
            || config.workers == 0
            || config.workers > 32
            || config.max_images == 0
            || config.max_images > 32
            || config.max_decoded_pixels == 0
            || config.max_decoded_pixels > 16_777_216
            || processor["patch_size"] != patch
            || processor["spatial_merge_size"] != merge
            || image["resample"] != 3
            || image["do_resize"] != true
            || image["do_rescale"] != true
            || image["do_normalize"] != true
            || image["do_convert_rgb"] != true
            || image["do_center_crop"] == true
            || image["do_pad"] == true
        {
            return Err(invalid(
                "Unsupported Mistral3/Pixtral processor configuration",
            ));
        }
        let mut mean = [0.; 3];
        let mut std = [0.; 3];
        for c in 0..3 {
            mean[c] = image["image_mean"][c]
                .as_f64()
                .ok_or_else(|| invalid("Missing Pixtral image mean"))? as f32;
            std[c] = image["image_std"][c]
                .as_f64()
                .ok_or_else(|| invalid("Missing Pixtral image standard deviation"))?
                as f32;
        }
        let scale = image["rescale_factor"]
            .as_f64()
            .ok_or_else(|| invalid("Missing image rescale factor"))? as f32;
        if !scale.is_finite()
            || scale <= 0.
            || mean.iter().any(|x| !x.is_finite())
            || std.iter().any(|x| !x.is_finite() || *x <= 0.)
        {
            return Err(invalid("Invalid Pixtral normalization"));
        }
        if (0..3).any(|c| {
            !((-mean[c]) / std[c]).is_finite() || !((255.0 * scale - mean[c]) / std[c]).is_finite()
        }) {
            return Err(invalid("Pixtral normalization overflows"));
        }
        let mut ids = [0; 3];
        for (i, key) in ["image_token", "image_break_token", "image_end_token"]
            .iter()
            .enumerate()
        {
            let name = processor[key]
                .as_str()
                .ok_or_else(|| invalid("Missing image token"))?;
            ids[i] = tokenizer
                .token_to_id(name)
                .ok_or_else(|| invalid("Missing tokenizer image token"))?;
        }
        if model["image_token_index"].as_u64() != Some(u64::from(ids[0])) {
            return Err(invalid("Image token IDs differ from model config"));
        }
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| invalid(e.to_string()))?;
        let chat = ChatProcessor::load(root)
            .map_err(invalid)?
            .ok_or_else(|| invalid("Mistral3 requires its native chat template"))?;
        let budget = MediaBudget::new(config.memory_budget_bytes)?;
        let resolver = MediaResolver::new(
            config.allowed_image_hosts.clone(),
            budget.clone(),
            config.max_image_bytes,
            config.download_concurrency,
            config.timeout,
        )?;
        let workers = Arc::new(Semaphore::new(config.workers));
        Ok(Self {
            state: Arc::new(State {
                chat,
                tokenizer,
                config,
                budget,
                max_tokens,
                patch,
                merge,
                edge,
                // Preserve the reference arithmetic order for every possible U8 value.
                normalized: std::array::from_fn(|c| {
                    std::array::from_fn(|v| (v as f32 * scale - mean[c]) / std[c])
                }),
                ids,
            }),
            resolver,
            workers,
        })
    }
    pub async fn prepare(
        &self,
        input: EncodingInput,
        truncate: bool,
        direction: TruncationDirection,
        prompt_name: Option<String>,
    ) -> Result<PreparedMultimodal> {
        if prompt_name.is_some() {
            return Err(invalid(
                "Mistral3 uses its native chat template; prompt_name is unsupported",
            ));
        }
        tokio::time::timeout(
            self.state.config.timeout,
            self.prepare_inner(input, truncate, direction),
        )
        .await
        .map_err(|_| invalid("Mistral3 preprocessing timed out"))?
    }
    async fn prepare_inner(
        &self,
        input: EncodingInput,
        truncate: bool,
        direction: TruncationDirection,
    ) -> Result<PreparedMultimodal> {
        if input.count_chars() > self.state.max_tokens.saturating_mul(250) {
            return Err(invalid("Conversation exceeds the character limit"));
        }
        let mut messages = match input {
            EncodingInput::Single(text) => vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text(text),
            }],
            EncodingInput::Messages(m) => m,
            _ => return Err(invalid("Mistral3 accepts text or native messages")),
        };
        validate(&messages, self.state.config.max_images)?;
        let mut resolved = Vec::new();
        for message in &mut messages {
            if let MessageContent::Parts(parts) = &mut message.content {
                for part in parts {
                    if let ContentPart::ImageUrl { image_url } = part {
                        resolved.push(
                            self.resolver
                                .resolve(std::mem::take(&mut image_url.url))
                                .await?,
                        );
                    }
                }
            }
        }
        let waiting = std::time::Instant::now();
        let worker = self
            .workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| invalid("Image workers closed"))?;
        metrics::histogram!("infinity_image_worker_wait_duration", "processor" => "mistral3")
            .record(waiting.elapsed().as_secs_f64());
        let state = self.state.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::task::spawn_blocking(move || {
            let _worker = worker;
            let started = std::time::Instant::now();
            let result =
                state.prepare_cpu(messages, resolved, truncate, direction, || tx.is_closed());
            metrics::histogram!("infinity_image_preprocessing_duration", "processor" => "mistral3")
                .record(started.elapsed().as_secs_f64());
            let _ = tx.send(result);
        });
        rx.await.map_err(|_| invalid("Image worker failed"))?
    }
}
fn validate(messages: &[Message], max_images: usize) -> Result<()> {
    if messages.is_empty() {
        return Err(invalid("Conversation must not be empty"));
    }
    let text = |s: &str| {
        if ["[IMG]", "[IMG_BREAK]", "[IMG_END]"]
            .iter()
            .any(|m| s.contains(m))
        {
            Err(invalid("Text must not contain reserved image markers"))
        } else {
            Ok(())
        }
    };
    let mut count = 0;
    for m in messages {
        match &m.content {
            MessageContent::Text(s) => text(s)?,
            MessageContent::Parts(parts) => {
                for p in parts {
                    match p {
                        ContentPart::Text { text: s } => text(s)?,
                        ContentPart::ImageUrl { image_url }
                            if matches!(m.role, MessageRole::User) =>
                        {
                            if !matches!(image_url.detail, None | Some(ImageDetail::Auto)) {
                                return Err(invalid("Image detail must be auto or omitted"));
                            }
                            count += 1;
                            if count > max_images {
                                return Err(invalid("Too many images"));
                            }
                        }
                        _ => return Err(invalid("Mistral3 supports text and user images only")),
                    }
                }
            }
        }
    }
    Ok(())
}
fn dimensions(width: u32, height: u32, edge: u32, factor: usize) -> Result<(u32, u32)> {
    if width == 0 || height == 0 || factor == 0 {
        return Err(invalid("Invalid image dimensions"));
    }
    let ratio = f64::from(width.max(height)) / f64::from(edge);
    let (w, h) = if ratio > 1. {
        (
            (f64::from(width) / ratio).floor() as usize,
            (f64::from(height) / ratio).floor() as usize,
        )
    } else {
        (width as usize, height as usize)
    };
    if w == 0 || h == 0 {
        return Err(invalid("Image aspect ratio produces an empty Pixtral grid"));
    }
    Ok((
        (w.div_ceil(factor) * factor) as u32,
        (h.div_ceil(factor) * factor) as u32,
    ))
}
impl State {
    fn prepare_cpu(
        &self,
        messages: Vec<Message>,
        resolved: Vec<ResolvedImage>,
        truncate: bool,
        direction: TruncationDirection,
        canceled: impl Fn() -> bool,
    ) -> Result<PreparedMultimodal> {
        let mut sizes = Vec::new();
        let mut bytes = 0usize;
        for image in &resolved {
            if canceled() {
                return Err(invalid("Image preprocessing canceled"));
            }
            let mut decoder =
                QwenImageProcessor::decoder(&image.bytes, self.config.max_decoded_pixels)
                    .map_err(invalid)?;
            let (mut w, mut h) = decoder.dimensions();
            let orientation = decoder
                .orientation()
                .map_err(|_| invalid("Invalid image orientation"))?;
            if matches!(
                orientation,
                image::metadata::Orientation::Rotate90
                    | image::metadata::Orientation::Rotate270
                    | image::metadata::Orientation::Rotate90FlipH
                    | image::metadata::Orientation::Rotate270FlipH
            ) {
                std::mem::swap(&mut w, &mut h);
            }
            let (rw, rh) = dimensions(w, h, self.edge, self.patch * self.merge)?;
            // Includes decoded/conversion buffers, resize intermediates, F32 patches and upload copies.
            bytes = bytes
                .checked_add(
                    (w as usize * h as usize) * 12
                        + (rw as usize * rh as usize) * 40
                        + (rw as usize * h as usize) * 4,
                )
                .ok_or_else(|| invalid("Image allocation overflow"))?;
            sizes.push((rw, rh));
        }
        let mut memory = self.budget.reserve(bytes)?;
        let prompt = self.chat.render_native(&messages, false).map_err(invalid)?;
        if prompt.chars().count() > self.max_tokens.saturating_mul(250) {
            return Err(invalid("Rendered conversation exceeds the character limit"));
        }
        // The template already contains BOS and turn markers.
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|_| invalid("Conversation tokenization failed"))?;
        let mut ids = Vec::new();
        let mut spans = Vec::new();
        let mut image_index = 0;
        for id in encoding.get_ids() {
            if *id != self.ids[0] {
                ids.push(*id);
                continue;
            }
            let (w, h) = *sizes
                .get(image_index)
                .ok_or_else(|| invalid("Image placeholder mismatch"))?;
            let nh = h as usize / (self.patch * self.merge);
            let nw = w as usize / (self.patch * self.merge);
            let total = nh * (nw + 1);
            if ids.len().saturating_add(total) > self.max_tokens {
                return Err(invalid(
                    "Image tokens exceed the input limit; multimodal inputs cannot be truncated",
                ));
            }
            if canceled() {
                return Err(invalid("Image preprocessing canceled"));
            }
            let rgb = resize_rgb(
                &resolved[image_index].bytes,
                w,
                h,
                self.config.max_decoded_pixels,
            )
            .map_err(invalid)?;
            let gh = h as usize / self.patch;
            let gw = w as usize / self.patch;
            let mut pixels = Vec::with_capacity(gh * gw * 3 * self.patch * self.patch);
            for y in 0..gh {
                for x in 0..gw {
                    for c in 0..3 {
                        for py in 0..self.patch {
                            for px in 0..self.patch {
                                let index =
                                    ((y * self.patch + py) * w as usize + x * self.patch + px) * 3
                                        + c;
                                pixels.push(self.normalized[c][rgb[index] as usize]);
                            }
                        }
                    }
                }
            }
            spans.push((
                ids.len(),
                Arc::new(ImagePatches {
                    pixels,
                    grid_thw: [1, gh, gw],
                    patch_dim: 3 * self.patch * self.patch,
                    merge_size: self.merge,
                }),
            ));
            for row in 0..nh {
                ids.resize(ids.len() + nw, self.ids[0]);
                ids.push(if row + 1 == nh {
                    self.ids[2]
                } else {
                    self.ids[1]
                });
            }
            image_index += 1;
        }
        if image_index != resolved.len() {
            return Err(invalid("Image count differs from template placeholders"));
        }
        if ids.len() > self.max_tokens {
            if !spans.is_empty() || !truncate {
                return Err(invalid("Input exceeds the configured token limit"));
            }
            match direction {
                TruncationDirection::Left => {
                    ids.drain(..ids.len() - self.max_tokens);
                }
                TruncationDirection::Right => ids.truncate(self.max_tokens),
            }
        }
        if ids.is_empty() {
            return Err(invalid("Empty model input"));
        }
        // Decode/resize buffers are gone. Keep charging patches and their upload
        // copies while queued/in flight, without retaining the CPU scratch budget.
        drop(resolved);
        let retained_bytes: usize = spans
            .iter()
            .map(|(_, image)| image.pixels.capacity() * std::mem::size_of::<f32>() * 2)
            .sum();
        let release = memory
            .num_permits()
            .saturating_sub(retained_bytes.div_ceil(1024));
        drop(memory.split(release));
        let positions = (0..ids.len() as u32).collect::<Vec<_>>();
        Ok(PreparedMultimodal {
            input_ids: ids,
            media: Arc::new(MultimodalEncoding {
                images: spans,
                audios: vec![],
                position_ids: std::array::from_fn(|_| positions.clone()),
                memory: Some(Arc::new(memory)),
            }),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pixtral_resize_ceil_and_budget_validation() {
        assert_eq!(dimensions(37, 65, 1540, 28).unwrap(), (56, 84));
        assert_eq!(dimensions(2000, 1000, 1540, 28).unwrap(), (1540, 784));
        assert!(dimensions(0, 10, 1540, 28).is_err());
        assert!(dimensions(16_000_000, 1, 1540, 28).is_err());
    }
    #[test]
    fn rejects_reserved_markers_and_unsupported_parts() {
        for json in [
            serde_json::json!([{"role":"user","content":"[IMG]"}]),
            serde_json::json!([{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"}}]}]),
            serde_json::json!([{"role":"user","content":[{"type":"video_url","video_url":{"url":"https://example.com/v"}}]}]),
        ] {
            let m: Vec<Message> = serde_json::from_value(json).unwrap();
            assert!(validate(&m, 4).is_err());
        }
    }
}
