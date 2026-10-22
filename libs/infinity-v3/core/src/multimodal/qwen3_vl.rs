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
    tokenizer: Tokenizer,
    fast_tokenizer: Option<crate::fast_tokenization::FastTokenizer>,
    images: QwenImageProcessor,
    budget: MediaBudget,
    config: MultimodalConfig,
    max_tokens: usize,
    image_id: u32,
    vision_start: u32,
    vision_end: u32,
}

impl Qwen3VlProcessor {
    pub fn load(
        root: &Path,
        mut tokenizer: Tokenizer,
        max_tokens: usize,
        config: MultimodalConfig,
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
            Some("qwen3_vl" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text")
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
        let fast_tokenizer = crate::fast_tokenization::load(&tokenizer);
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
        let prompt = self.chat.render_native(&messages, true).map_err(invalid)?;
        if prompt.chars().count() > self.max_tokens.saturating_mul(250) {
            return Err(invalid("Rendered conversation exceeds the character limit"));
        }
        let mut encoding = match &self.fast_tokenizer {
            Some(fast) => crate::fast_tokenization::encode(fast, &self.tokenizer, &prompt, true)
                .or_else(|_| self.tokenizer.encode(prompt.as_str(), true)),
            None => self.tokenizer.encode(prompt.as_str(), true),
        }
        .map_err(|_| invalid("Native conversation tokenization failed"))?;
        if plans.is_empty() && truncate && encoding.len() > self.max_tokens {
            // Let the tokenizer truncate before its post-processor adds the pooled
            // terminal token. Slicing finalized IDs would discard that token.
            let mut tokenizer = self.tokenizer.clone();
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
    use serde_json::json;

    #[test]
    fn validates_all_parts_before_media_resolution() {
        let valid = json!([{"role":"user","content":[
            {"type":"text","text":"Compare"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}
        ]},{"role":"assistant","content":"A picture"}]);
        let messages: Vec<Message> = serde_json::from_value(valid.clone()).unwrap();
        assert!(validate_messages(&messages, 1).is_ok());
        assert!(validate_messages(&messages, 0).is_err());
        for count in [16, 17] {
            let mut request = valid.clone();
            request[0]["content"] = json!(vec![valid[0]["content"][1].clone(); count]);
            let messages = serde_json::from_value::<Vec<Message>>(request).unwrap();
            assert_eq!(
                validate_messages(&messages, MultimodalConfig::default().max_images).is_ok(),
                count == 16
            );
        }
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
