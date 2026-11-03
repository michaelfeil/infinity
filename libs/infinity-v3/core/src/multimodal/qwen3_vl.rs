//! Async media resolution followed by bounded, model-owned CPU preparation.
use super::{
    image::QwenImageProcessor,
    media::{MediaBudget, MediaResolver, ResolvedImage},
    positions::image_positions,
};
use crate::{
    chat::ChatProcessor,
    input::{ContentPart, ImageDetail, Message, MessageContent, MessageRole},
    tokenization::EncodingInput,
    TextEmbeddingsError,
};
use infinity_v3_backend::{ImagePatches, MultimodalEncoding};
use std::{path::Path, sync::Arc, time::Duration};
use tokenizers::{Tokenizer, TruncationDirection};
use tokio::sync::{oneshot, Semaphore};

type Result<T> = std::result::Result<T, TextEmbeddingsError>;
fn invalid(message: impl Into<String>) -> TextEmbeddingsError {
    TextEmbeddingsError::Tokenizer(message.into().into())
}

#[derive(Debug, Clone)]
pub struct MultimodalConfig {
    pub allowed_image_hosts: Vec<String>,
    pub max_images: usize,
    pub max_image_bytes: usize,
    pub max_decoded_pixels: u64,
    pub memory_budget_bytes: usize,
    pub workers: usize,
    pub download_concurrency: usize,
    pub timeout: Duration,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            allowed_image_hosts: super::media::default_image_allowed_hosts(),
            max_images: 16,
            max_image_bytes: 20 * 1024 * 1024,
            max_decoded_pixels: 16_777_216,
            memory_budget_bytes: 512 * 1024 * 1024,
            workers: 2,
            download_concurrency: 8,
            timeout: Duration::from_secs(30),
        }
    }
}

impl MultimodalConfig {
    /// Validate limits and host patterns without loading model assets or images.
    pub fn validate(&self) -> Result<()> {
        if self.max_images == 0
            || self.max_images > 32
            || self.max_decoded_pixels == 0
            || self.max_decoded_pixels > 16_777_216
            || self.workers == 0
            || self.workers > 32
        {
            return Err(invalid("Invalid multimodal processing limits"));
        }
        let budget = MediaBudget::new(self.memory_budget_bytes)?;
        MediaResolver::new(
            self.allowed_image_hosts.clone(),
            budget,
            self.max_image_bytes,
            self.download_concurrency,
            self.timeout,
        )?;
        Ok(())
    }
}

pub struct PreparedQwenImages {
    pub images: Vec<Arc<ImagePatches>>,
    pub memory: Arc<tokio::sync::OwnedSemaphorePermit>,
}

pub struct PreparedMultimodal {
    pub input_ids: Vec<u32>,
    pub media: Arc<MultimodalEncoding>,
}

#[derive(Clone)]
pub struct Qwen3VlProcessor {
    state: Arc<ProcessorState>,
    resolver: MediaResolver,
    workers: Arc<Semaphore>,
}

impl std::fmt::Debug for Qwen3VlProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3VlProcessor").finish_non_exhaustive()
    }
}

struct ProcessorState {
    chat: ChatProcessor,
    tokenizer: Arc<Tokenizer>,
    fast_tokenizer: crate::fast_tokenization::BackgroundFastTokenizer,
    images: QwenImageProcessor,
    budget: MediaBudget,
    config: MultimodalConfig,
    max_tokens: usize,
    image_id: u32,
    vision_start: u32,
    vision_end: u32,
    colqwen: bool,
    raw_text: bool,
}

impl Qwen3VlProcessor {
    pub fn load(
        root: &Path,
        tokenizer: Tokenizer,
        max_tokens: usize,
        config: MultimodalConfig,
    ) -> Result<Self> {
        Self::load_with_pooled_text_acceleration(root, tokenizer, max_tokens, config, true)
    }

    /// Load an image-only processor for callers that never encode text through
    /// this processor, such as OneJev. Validation and image preparation are
    /// identical to [`Self::load`], but the pooled-text Basetenkenizer is not
    /// constructed.
    pub fn load_images_only(
        root: &Path,
        tokenizer: Tokenizer,
        max_tokens: usize,
        config: MultimodalConfig,
    ) -> Result<Self> {
        Self::load_with_pooled_text_acceleration(root, tokenizer, max_tokens, config, false)
    }

    fn load_with_pooled_text_acceleration(
        root: &Path,
        mut tokenizer: Tokenizer,
        max_tokens: usize,
        config: MultimodalConfig,
        accelerate_pooled_text: bool,
    ) -> Result<Self> {
        if config.max_images == 0
            || config.max_images > 32
            || config.max_decoded_pixels == 0
            || config.max_decoded_pixels > 16_777_216
            || config.workers == 0
            || config.workers > 32
            || max_tokens == 0
            || max_tokens > u32::MAX as usize
        {
            return Err(invalid("Invalid multimodal processing limits"));
        }
        let model: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("config.json")).map_err(|e| invalid(e.to_string()))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        if !matches!(
            model["model_type"].as_str(),
            Some(
                "colqwen3"
                    | "qwen3_vl_nemotron_embed"
                    | "qwen3_vl"
                    | "qwen3_5"
                    | "qwen3_5_text"
                    | "qwen3_5_moe"
                    | "qwen3_5_moe_text"
            )
        ) {
            return Err(invalid("Image processor requires a Qwen vision checkpoint"));
        }
        let token = |name: &str, config_name: &str| -> Result<u32> {
            let id = tokenizer
                .token_to_id(name)
                .ok_or_else(|| invalid("Checkpoint is missing required vision tokens"))?;
            if model[config_name].as_u64() != Some(u64::from(id)) {
                return Err(invalid(
                    "Tokenizer vision IDs do not match model configuration",
                ));
            }
            Ok(id)
        };
        let image_id = token("<|image_pad|>", "image_token_id")?;
        let vision_start = token("<|vision_start|>", "vision_start_token_id")?;
        let vision_end = token("<|vision_end|>", "vision_end_token_id")?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| invalid(e.to_string()))?;
        let chat = ChatProcessor::load(root)
            .map_err(invalid)?
            .ok_or_else(|| invalid("Qwen3-VL requires the checkpoint's native chat template"))?;
        let images = QwenImageProcessor::load(root).map_err(invalid)?;
        let budget = MediaBudget::new(config.memory_budget_bytes)?;
        let resolver = MediaResolver::new(
            config.allowed_image_hosts.clone(),
            budget.clone(),
            config.max_image_bytes,
            config.download_concurrency,
            config.timeout,
        )?;
        let workers = Arc::new(Semaphore::new(config.workers));
        let tokenizer = Arc::new(tokenizer);
        let fast_tokenizer = if accelerate_pooled_text {
            crate::fast_tokenization::BackgroundFastTokenizer::new(tokenizer.clone())
        } else {
            crate::fast_tokenization::BackgroundFastTokenizer::disabled()
        };
        Ok(Self {
            state: Arc::new(ProcessorState {
                chat,
                tokenizer,
                fast_tokenizer,
                images,
                budget,
                config,
                max_tokens,
                image_id,
                vision_start,
                vision_end,
                colqwen: model["model_type"] == "colqwen3",
                raw_text: matches!(
                    model["model_type"].as_str(),
                    Some("colqwen3" | "qwen3_vl_nemotron_embed")
                ),
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
                "Qwen3-VL uses its native template; prompt_name is unsupported",
            ));
        }
        tokio::time::timeout(
            self.state.config.timeout,
            self.prepare_inner(input, truncate, direction),
        )
        .await
        .map_err(|_| invalid("Multimodal preprocessing timed out"))?
    }

    pub fn accepts_raw_text(&self) -> bool {
        self.state.raw_text
    }

    pub fn max_images(&self) -> usize {
        self.state.config.max_images
    }

    /// Prepare shared image patches independently of a task's prompt template.
    pub async fn prepare_images(
        &self,
        sources: Vec<String>,
        max_image_tokens: usize,
    ) -> Result<PreparedQwenImages> {
        if sources.is_empty()
            || sources.len() > self.state.config.max_images
            || max_image_tokens == 0
        {
            return Err(invalid("Invalid number of images"));
        }
        tokio::time::timeout(self.state.config.timeout, async {
            let mut resolved = Vec::with_capacity(sources.len());
            for source in sources {
                resolved.push(self.resolver.resolve(source).await?);
            }
            let worker = self
                .workers
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| invalid("Image processing workers closed"))?;
            let state = self.state.clone();
            let (tx, rx) = oneshot::channel();
            tokio::task::spawn_blocking(move || {
                let _worker = worker;
                let result = (|| {
                    let mut plans = Vec::with_capacity(resolved.len());
                    let mut bytes = 0usize;
                    for image in &resolved {
                        if tx.is_closed() {
                            return Err(invalid("Image preprocessing canceled"));
                        }
                        let plan = state
                            .images
                            .inspect(&image.bytes, state.config.max_decoded_pixels)
                            .map_err(invalid)?;
                        if plan.token_count > max_image_tokens {
                            return Err(invalid("Image exceeds vision token limit"));
                        }
                        bytes = bytes
                            .checked_add(plan.memory_bytes)
                            .ok_or_else(|| invalid("Image allocation is too large"))?;
                        plans.push(plan);
                    }
                    let memory = Arc::new(state.budget.reserve(bytes)?);
                    let mut images = Vec::with_capacity(plans.len());
                    for (image, plan) in resolved.into_iter().zip(plans) {
                        if tx.is_closed() {
                            return Err(invalid("Image preprocessing canceled"));
                        }
                        images.push(Arc::new(
                            state
                                .images
                                .prepare(&image.bytes, &plan, state.config.max_decoded_pixels)
                                .map_err(invalid)?,
                        ));
                    }
                    Ok(PreparedQwenImages { images, memory })
                })();
                let _ = tx.send(result);
            });
            rx.await
                .map_err(|_| invalid("Image processing worker failed"))?
        })
        .await
        .map_err(|_| invalid("Image preprocessing timed out"))?
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
            EncodingInput::Messages(messages) => messages,
            _ => {
                return Err(invalid(
                    "Qwen3-VL accepts text or native messages, not raw token IDs or pairs",
                ))
            }
        };
        validate_messages(&messages, self.state.config.max_images)?;
        let mut resolved = Vec::new();
        for message in &mut messages {
            if let MessageContent::Parts(parts) = &mut message.content {
                for part in parts {
                    if let ContentPart::ImageUrl { image_url } = part {
                        // Remove the source as it is resolved: templates see the ordered image part,
                        // but cannot accidentally emit/log a signed URL or a large base64 string.
                        resolved.push(
                            self.resolver
                                .resolve(std::mem::take(&mut image_url.url))
                                .await?,
                        );
                    }
                }
            }
        }
        // HTTP admission and media reservations already bound pending work. Wait
        // for a CPU slot so an ordinary text batch can exceed the worker count.
        let worker = self
            .workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| invalid("Image processing workers closed"))?;
        let state = self.state.clone();
        let (sender, receiver) = oneshot::channel();
        tokio::task::spawn_blocking(move || {
            let _worker = worker;
            let result = state.prepare_cpu(messages, resolved, truncate, direction, || {
                sender.is_closed()
            });
            let _ = sender.send(result);
        });
        receiver
            .await
            .map_err(|_| invalid("Image processing worker failed"))?
    }
}

fn validate_messages(messages: &[Message], max_images: usize) -> Result<()> {
    if messages.is_empty() {
        return Err(invalid("Conversation must not be empty"));
    }
    let mut count = 0;
    let text = |value: &str| -> Result<()> {
        if [
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|video_pad|>",
        ]
        .iter()
        .any(|marker| value.contains(marker))
        {
            return Err(invalid("Text must not contain reserved vision markers"));
        }
        Ok(())
    };
    for message in messages {
        if !matches!(message.role, MessageRole::User | MessageRole::Assistant) {
            return Err(invalid("Qwen3-VL accepts user and assistant messages"));
        }
        match &message.content {
            MessageContent::Text(value) => text(value)?,
            MessageContent::Parts(parts) => {
                for part in parts {
                    match part {
                        ContentPart::Text { text: value } => text(value)?,
                        ContentPart::ImageUrl { image_url }
                            if matches!(message.role, MessageRole::User) =>
                        {
                            if !matches!(image_url.detail, None | Some(ImageDetail::Auto)) {
                                return Err(invalid(
                                    "Qwen3-VL image detail must be auto or omitted",
                                ));
                            }
                            count += 1;
                            if count > max_images {
                                return Err(invalid("Too many images in one conversation"));
                            }
                        }
                        _ => return Err(invalid(
                            "Only user image parts and text are supported by this model processor",
                        )),
                    }
                }
            }
        }
    }
    Ok(())
}

impl ProcessorState {
    fn prepare_cpu(
        &self,
        messages: Vec<Message>,
        resolved: Vec<ResolvedImage>,
        truncate: bool,
        direction: TruncationDirection,
        canceled: impl Fn() -> bool,
    ) -> Result<PreparedMultimodal> {
        let check_cancel = || {
            if canceled() {
                Err(invalid("Image preprocessing canceled"))
            } else {
                Ok(())
            }
        };
        check_cancel()?;
        let mut plans = Vec::with_capacity(resolved.len());
        let mut memory_bytes = 0usize;
        for image in &resolved {
            check_cancel()?;
            let plan = self
                .images
                .inspect(&image.bytes, self.config.max_decoded_pixels)
                .map_err(invalid)?;
            memory_bytes = memory_bytes
                .checked_add(plan.memory_bytes)
                .ok_or_else(|| invalid("Image allocation is too large"))?;
            plans.push(plan);
        }
        // Qwen3-VL-Embedding pools the terminal token added by its tokenizer's
        // post-processor after the assistant prefix (add_special_tokens=true).
        let prompt = if self.colqwen {
            // ColQwen3 document prompts end at the user turn, followed by EOS.
            format!(
                "{}<|endoftext|>",
                self.chat
                    .render_native(&messages, false)
                    .map_err(invalid)?
                    .trim_end_matches('\n')
            )
        } else {
            self.chat.render_native(&messages, true).map_err(invalid)?
        };
        if prompt.chars().count() > self.max_tokens.saturating_mul(250) {
            return Err(invalid("Rendered conversation exceeds the character limit"));
        }
        let mut encoding = match self.fast_tokenizer.get() {
            Some(fast) => {
                crate::fast_tokenization::encode(fast, self.tokenizer.as_ref(), &prompt, true)
                    .or_else(|_| self.tokenizer.encode(prompt.as_str(), true))
            }
            None => self.tokenizer.encode(prompt.as_str(), true),
        }
        .map_err(|_| invalid("Native conversation tokenization failed"))?;
        if plans.is_empty() && truncate && encoding.len() > self.max_tokens {
            // Let the tokenizer truncate before its post-processor adds the pooled
            // terminal token. Slicing finalized IDs would discard that token.
            let mut tokenizer = self.tokenizer.as_ref().clone();
            tokenizer
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length: self.max_tokens,
                    direction,
                    ..Default::default()
                }))
                .map_err(|_| invalid("Invalid text truncation limit"))?;
            encoding = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|_| invalid("Native conversation tokenization failed"))?;
        }
        let raw = encoding.get_ids();
        let markers = raw.iter().filter(|&&id| id == self.image_id).count();
        if markers != plans.len() {
            return Err(invalid("Image marker count does not match message images"));
        }
        let expanded =
            raw.len() - markers + plans.iter().map(|plan| plan.token_count).sum::<usize>();
        if expanded > self.max_tokens && (!truncate || !plans.is_empty()) {
            return Err(TextEmbeddingsError::Validation(
                "Input exceeds the token limit; image spans cannot be truncated".into(),
            ));
        }
        let memory = self.budget.reserve(memory_bytes)?;
        let mut images = Vec::with_capacity(resolved.len());
        for (image, plan) in resolved.into_iter().zip(&plans) {
            check_cancel()?;
            images.push(
                self.images
                    .prepare(&image.bytes, plan, self.config.max_decoded_pixels)
                    .map_err(invalid)?,
            );
        }
        check_cancel()?;
        let mut input_ids = Vec::with_capacity(expanded);
        let mut spans = Vec::with_capacity(images.len());
        let mut images = images.into_iter();
        for (index, &id) in raw.iter().enumerate() {
            if id == self.image_id {
                if index == 0
                    || raw[index - 1] != self.vision_start
                    || raw.get(index + 1) != Some(&self.vision_end)
                {
                    return Err(invalid(
                        "Unexpected image marker layout in the model template",
                    ));
                }
                let image = images.next().unwrap();
                let start = input_ids.len();
                input_ids.resize(start + image.token_count(), id);
                spans.push((start, Arc::new(image)));
            } else {
                input_ids.push(id);
            }
        }
        if input_ids.is_empty() {
            return Err(invalid("The model template produced no input tokens"));
        }
        let position_ids = image_positions(
            input_ids.len(),
            &spans
                .iter()
                .map(|(start, image)| (*start, image.as_ref()))
                .collect::<Vec<_>>(),
        )
        .map_err(invalid)?;
        check_cancel()?;
        Ok(PreparedMultimodal {
            input_ids,
            media: Arc::new(MultimodalEncoding {
                audios: vec![],
                images: spans,
                position_ids,
                memory: Some(Arc::new(memory)),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde_json::json;
    use std::io::Cursor;
    use tokenizers::AddedToken;

    struct ImageFixture {
        _directory: tempfile::TempDir,
        root: std::path::PathBuf,
        tokenizer: Tokenizer,
    }

    fn image_fixture(image_id_offset: i64) -> ImageFixture {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().to_path_buf();
        let mut tokenizer = crate::fast_tokenization::test_tokenizer();
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|image_pad|>", true),
            AddedToken::from("<|vision_start|>", true),
            AddedToken::from("<|vision_end|>", true),
        ]);
        let image_id = i64::from(tokenizer.token_to_id("<|image_pad|>").unwrap());
        let vision_start = tokenizer.token_to_id("<|vision_start|>").unwrap();
        let vision_end = tokenizer.token_to_id("<|vision_end|>").unwrap();
        std::fs::write(
            root.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();
        std::fs::write(
            root.join("chat_template.jinja"),
            "{{ messages[0].content }}",
        )
        .unwrap();
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&json!({
                "model_type": "qwen3_5",
                "image_token_id": u64::try_from(image_id + image_id_offset).unwrap(),
                "vision_start_token_id": vision_start,
                "vision_end_token_id": vision_end,
                "vision_config": {},
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            root.join("preprocessor_config.json"),
            r#"{"patch_size":16,"temporal_patch_size":2,"merge_size":2,"min_pixels":4096,"max_pixels":1310720,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"rescale_factor":0.00392156862745098,"resample":3,"do_resize":true,"do_rescale":true,"do_normalize":true,"do_convert_rgb":true}"#,
        )
        .unwrap();
        ImageFixture {
            _directory: directory,
            root,
            tokenizer,
        }
    }

    fn png_data_url(width: u32, height: u32) -> String {
        let image = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([
                ((x * 17 + y * 3) % 256) as u8,
                ((x * 5 + y * 29) % 256) as u8,
                (((x ^ y) * 23) % 256) as u8,
            ])
        });
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        format!(
            "data:image/png;base64,{}",
            STANDARD.encode(bytes.into_inner())
        )
    }

    fn text_messages() -> Vec<Message> {
        serde_json::from_value(json!([
            {"role":"user","content":"Compare the weather in Tokyo and Paris."},
            {"role":"assistant","content":"It is raining in Tokyo."},
            {"role":"user","content":"Summarize."}
        ]))
        .unwrap()
    }

    fn assert_same_images(regular: &PreparedQwenImages, images_only: &PreparedQwenImages) {
        assert_eq!(regular.images.len(), images_only.images.len());
        for (regular, images_only) in regular.images.iter().zip(&images_only.images) {
            assert_eq!(regular.token_count(), images_only.token_count());
            assert_eq!(regular.grid_thw, images_only.grid_thw);
            assert_eq!(regular.patch_dim, images_only.patch_dim);
            assert_eq!(regular.merge_size, images_only.merge_size);
            assert_eq!(regular.pixels, images_only.pixels);
        }
    }

    #[tokio::test]
    async fn images_only_matches_regular_preparation_and_validation() {
        let fixture = image_fixture(0);
        let config = MultimodalConfig::default();
        let regular = Qwen3VlProcessor::load(
            &fixture.root,
            fixture.tokenizer.clone(),
            8192,
            config.clone(),
        )
        .unwrap();
        let images_only = Qwen3VlProcessor::load_images_only(
            &fixture.root,
            fixture.tokenizer.clone(),
            8192,
            config.clone(),
        )
        .unwrap();
        assert!(images_only.state.fast_tokenizer.get().is_none());

        let source = png_data_url(64, 96);
        let regular_images = regular
            .prepare_images(vec![source.clone()], 4096)
            .await
            .unwrap();
        let only_images = images_only
            .prepare_images(vec![source.clone()], 4096)
            .await
            .unwrap();
        assert_same_images(&regular_images, &only_images);
        let token_count = regular_images.images[0].token_count();
        assert!(token_count > 0);

        let regular_limit = regular
            .prepare_images(vec![source.clone()], token_count - 1)
            .await
            .err()
            .expect("regular processor accepted an over-budget image")
            .to_string();
        let images_only_limit = images_only
            .prepare_images(vec![source.clone()], token_count - 1)
            .await
            .err()
            .expect("image-only processor accepted an over-budget image")
            .to_string();
        assert_eq!(regular_limit, images_only_limit);
        assert!(regular_limit.contains("vision token limit"));

        let mut tight = config.clone();
        tight.memory_budget_bytes = 4096;
        let regular_tight = Qwen3VlProcessor::load(
            &fixture.root,
            fixture.tokenizer.clone(),
            8192,
            tight.clone(),
        )
        .unwrap();
        let images_only_tight =
            Qwen3VlProcessor::load_images_only(&fixture.root, fixture.tokenizer, 8192, tight)
                .unwrap();
        let regular_budget = regular_tight
            .prepare_images(vec![source.clone()], 4096)
            .await
            .err()
            .expect("regular processor exceeded its memory budget")
            .to_string();
        let images_only_budget = images_only_tight
            .prepare_images(vec![source], 4096)
            .await
            .err()
            .expect("image-only processor exceeded its memory budget")
            .to_string();
        assert_eq!(regular_budget, images_only_budget);

        let invalid = image_fixture(1);
        let regular_error = Qwen3VlProcessor::load(
            &invalid.root,
            invalid.tokenizer.clone(),
            8192,
            config.clone(),
        )
        .unwrap_err()
        .to_string();
        let images_only_error =
            Qwen3VlProcessor::load_images_only(&invalid.root, invalid.tokenizer, 8192, config)
                .unwrap_err()
                .to_string();
        assert_eq!(regular_error, images_only_error);
        assert!(regular_error.contains("vision IDs"));
    }

    #[tokio::test]
    async fn fallback_and_accelerated_processing_match_images_only() {
        let fixture = image_fixture(0);
        let config = MultimodalConfig::default();
        let regular = Qwen3VlProcessor::load(
            &fixture.root,
            fixture.tokenizer.clone(),
            8192,
            config.clone(),
        )
        .unwrap();
        let images_only = Qwen3VlProcessor::load_images_only(
            &fixture.root,
            fixture.tokenizer.clone(),
            8192,
            config,
        )
        .unwrap();
        assert!(images_only.state.fast_tokenizer.get().is_none());

        let regular_text = regular
            .prepare(
                EncodingInput::Messages(text_messages()),
                true,
                TruncationDirection::Right,
                None,
            )
            .await
            .unwrap();
        let fallback_text = images_only
            .prepare(
                EncodingInput::Messages(text_messages()),
                true,
                TruncationDirection::Right,
                None,
            )
            .await
            .unwrap();
        assert_eq!(regular_text.input_ids, fallback_text.input_ids);
        assert!(regular_text.media.images.is_empty());
        assert!(fallback_text.media.images.is_empty());

        let source = png_data_url(64, 96);
        let regular_images = regular
            .prepare_images(vec![source.clone()], 4096)
            .await
            .unwrap();
        let fallback_images = images_only
            .prepare_images(vec![source], 4096)
            .await
            .unwrap();
        assert_same_images(&regular_images, &fallback_images);
    }

    #[tokio::test]
    async fn pending_accelerator_matches_hf_text_and_visual_processing() {
        let fixture = image_fixture(0);
        let config = MultimodalConfig::default();
        let fallback = Qwen3VlProcessor::load_images_only(
            &fixture.root,
            fixture.tokenizer.clone(),
            8192,
            config.clone(),
        )
        .unwrap();
        let mut pending = Qwen3VlProcessor::load_images_only(
            &fixture.root,
            fixture.tokenizer.clone(),
            8192,
            config,
        )
        .unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let state = Arc::get_mut(&mut pending.state).expect("fresh processor state");
        state.fast_tokenizer = crate::fast_tokenization::BackgroundFastTokenizer::new_gated(
            state.tokenizer.clone(),
            started_tx,
            release_rx,
        );
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("background tokenizer builder did not start");
        assert!(pending.state.fast_tokenizer.get().is_none());

        let pending_text = pending
            .prepare(
                EncodingInput::Messages(text_messages()),
                true,
                TruncationDirection::Right,
                None,
            )
            .await
            .unwrap();
        let fallback_text = fallback
            .prepare(
                EncodingInput::Messages(text_messages()),
                true,
                TruncationDirection::Right,
                None,
            )
            .await
            .unwrap();
        assert_eq!(pending_text.input_ids, fallback_text.input_ids);
        assert!(pending_text.media.images.is_empty());
        assert!(fallback_text.media.images.is_empty());

        let source = png_data_url(64, 96);
        let pending_images = pending
            .prepare_images(vec![source.clone()], 4096)
            .await
            .unwrap();
        let fallback_images = fallback.prepare_images(vec![source], 4096).await.unwrap();
        assert_same_images(&pending_images, &fallback_images);

        release_tx.send(()).expect("release tokenizer builder");
        drop(pending);
    }

    #[test]
    fn validates_all_parts_before_media_resolution() {
        let valid = json!([{"role":"user","content":[
            {"type":"text","text":"Compare"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}
        ]},{"role":"assistant","content":"A picture"}]);
        let messages: Vec<Message> = serde_json::from_value(valid.clone()).unwrap();
        assert!(validate_messages(&messages, 1).is_ok());
        assert!(validate_messages(&messages, 0).is_err());
        for invalid in [
            json!([]),
            json!([{"role":"system","content":"hi"}]),
            json!([{"role":"user","content":"<|image_pad|>"}]),
            json!([{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"https://example.com/a"}}]}]),
            json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a","detail":"high"}}]}]),
            json!([{"role":"user","content":[{"type":"video_url","video_url":{"url":"https://example.com/a"}}]}]),
        ] {
            let messages: Vec<Message> = serde_json::from_value(invalid).unwrap();
            assert!(validate_messages(&messages, 4).is_err());
        }
    }

    /// Regenerate fixtures with scripts/qwen3-vl-preprocessing-reference.py.
    #[tokio::test]
    #[ignore = "requires the pinned tokenizer and generated Transformers reference fixtures"]
    async fn native_processor_matches_reference() {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let root = std::path::PathBuf::from(std::env::var("QWEN3_VL_CHECKPOINT").unwrap());
        let fixtures = std::path::PathBuf::from(std::env::var("QWEN3_VL_FIXTURES").unwrap());
        let load = |max_tokens| {
            Qwen3VlProcessor::load(
                &root,
                Tokenizer::from_file(root.join("tokenizer.json")).unwrap(),
                max_tokens,
                MultimodalConfig::default(),
            )
            .unwrap()
        };
        let processor = load(8192);
        for case in ["one_image", "two_images"] {
            let reference: serde_json::Value = serde_json::from_slice(
                &std::fs::read(fixtures.join(format!("{case}.sequence.json"))).unwrap(),
            )
            .unwrap();
            let mut content = vec![json!({"type":"text","text":"Compare these images: "})];
            for (index, name) in reference["image_names"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                let bytes = std::fs::read(fixtures.join(format!("{}.png", name.as_str().unwrap())))
                    .unwrap();
                content.push(json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}",STANDARD.encode(bytes))}}));
                content.push(json!({"type":"text","text":format!(" Image {}. ",index+1)}));
            }
            let messages: Vec<Message> = serde_json::from_value(json!([
                {"role":"user","content":content},
                {"role":"assistant","content":"I see the images."},
                {"role":"user","content":"Represent their content."}
            ]))
            .unwrap();
            let prepared = processor
                .prepare(
                    EncodingInput::Messages(messages.clone()),
                    false,
                    TruncationDirection::Right,
                    None,
                )
                .await
                .unwrap();
            let expected: Vec<u32> =
                serde_json::from_value(reference["input_ids"][0].clone()).unwrap();
            assert_eq!(prepared.input_ids, expected, "{case}: token IDs");
            for axis in 0..3 {
                let expected: Vec<u32> =
                    serde_json::from_value(reference["position_ids"][axis][0].clone()).unwrap();
                assert_eq!(
                    prepared.media.position_ids[axis], expected,
                    "{case}: positions axis {axis}"
                );
            }
            for ((_, image), name) in prepared
                .media
                .images
                .iter()
                .zip(reference["image_names"].as_array().unwrap())
            {
                let bytes =
                    std::fs::read(fixtures.join(format!("{}.pixels.f32", name.as_str().unwrap())))
                        .unwrap();
                let expected: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                assert_eq!(image.pixels, expected, "{case}: image patches");
            }
            // Media cannot be silently cut, even when text truncation is requested.
            let limited = load(expected.len() - 1);
            assert!(limited
                .prepare(
                    EncodingInput::Messages(messages),
                    true,
                    TruncationDirection::Right,
                    None
                )
                .await
                .is_err());
            for direction in [TruncationDirection::Left, TruncationDirection::Right] {
                let text = limited
                    .prepare(
                        EncodingInput::Single("hello ".repeat(200)),
                        true,
                        direction,
                        None,
                    )
                    .await
                    .unwrap();
                assert_eq!(text.input_ids.len(), expected.len() - 1);
                assert_eq!(text.input_ids.last(), Some(&151643));
            }
            let memory = processor.state.budget.clone();
            let held = prepared.media.clone();
            drop(prepared);
            assert!(memory
                .reserve(processor.state.config.memory_budget_bytes)
                .is_err());
            drop(held);
            assert!(memory
                .reserve(processor.state.config.memory_budget_bytes)
                .is_ok());
        }
    }
}
