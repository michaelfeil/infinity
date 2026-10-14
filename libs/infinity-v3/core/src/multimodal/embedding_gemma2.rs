//! EmbeddingGemma 2's interleaved text/image/video/audio input preparation.
use super::{
    gemma4::Gemma4ImageProcessor,
    media::{MediaBudget, MediaResolver},
    MultimodalConfig, PreparedMultimodal,
};
use crate::{
    input::{AudioFormat, ContentPart, ImageDetail, Message, MessageContent, MessageRole},
    TextEmbeddingsError,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use infinity_v3_backend::{AudioFeatures, ImagePatches, MultimodalEncoding};
use std::{collections::HashMap, path::Path, process::Stdio, sync::Arc};
use tokenizers::{Tokenizer, TruncationDirection};
use tokio::{
    io::AsyncReadExt,
    process::Command,
    sync::{OwnedSemaphorePermit, Semaphore},
};

type Result<T> = std::result::Result<T, TextEmbeddingsError>;
fn invalid(message: impl Into<String>) -> TextEmbeddingsError {
    TextEmbeddingsError::Validation(message.into())
}

#[derive(Clone)]
pub struct EmbeddingGemma2Processor {
    tokenizer: Arc<Tokenizer>,
    images: Gemma4ImageProcessor,
    frames: Gemma4ImageProcessor,
    resolver: MediaResolver,
    budget: MediaBudget,
    workers: Arc<Semaphore>,
    config: MultimodalConfig,
    max_tokens: usize,
    ids: [u32; 7], // image, video, audio, begin image, end image, begin audio, end audio
    prompts: HashMap<String, String>,
    default_prompt: Option<String>,
    video_fps: f64,
    max_frames: usize,
}

impl std::fmt::Debug for EmbeddingGemma2Processor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGemma2Processor")
            .finish_non_exhaustive()
    }
}

enum PreparedItem {
    Images(Vec<Arc<ImagePatches>>, bool),
    Audio(Arc<AudioFeatures>),
}

impl EmbeddingGemma2Processor {
    pub fn load(
        root: &Path,
        mut tokenizer: Tokenizer,
        max_tokens: usize,
        config: MultimodalConfig,
        prompts: Option<HashMap<String, String>>,
        default_prompt: Option<String>,
    ) -> Result<Self> {
        let processor: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("processor_config.json"))
                .map_err(|e| invalid(e.to_string()))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let model: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("config.json")).map_err(|e| invalid(e.to_string()))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let video = &processor["video_processor"];
        let audio = &processor["feature_extractor"];
        // These are the released model's feature semantics; reject incompatible exports.
        if audio["sampling_rate"] != 16000
            || audio["feature_size"] != 128
            || audio["frame_length"] != 320
            || audio["hop_length"] != 160
            || audio["fft_length"] != 512
            || audio["preemphasis"] != 0.0
            || audio["dither"] != 0.0
            || audio["input_scale_factor"] != 1.0
            || audio["min_frequency"] != 0.0
            || audio["max_frequency"] != 8000.0
            || audio["mel_floor"] != 0.001
            || !audio["per_bin_mean"].is_null()
            || !audio["per_bin_stddev"].is_null()
            || audio["fft_overdrive"] != false
            || video["add_timestamps"] != false
            || video["overflow_strategy"] != "uniform"
            || video["do_sample_frames"] != true
            || max_tokens == 0
        {
            return Err(invalid(
                "Unsupported EmbeddingGemma2 media processor configuration",
            ));
        }
        let video_fps = video["fps"]
            .as_f64()
            .filter(|fps| fps.is_finite() && *fps > 0.0)
            .ok_or_else(|| invalid("Invalid video sampling rate"))?;
        let max_frames = video["max_frames"]
            .as_u64()
            .filter(|n| *n > 0 && *n <= 32)
            .ok_or_else(|| invalid("Invalid video frame limit"))? as usize;
        let token_names = [
            "<|image|>",
            "<|video|>",
            "<|audio|>",
            "<|image>",
            "<image|>",
            "<|audio>",
            "<audio|>",
        ];
        let config_names = [
            "image_token_id",
            "video_token_id",
            "audio_token_id",
            "boi_token_id",
            "eoi_token_id",
            "boa_token_id",
            "eoa_token_index",
        ];
        let mut ids = [0; 7];
        for (i, name) in token_names.iter().enumerate() {
            ids[i] = tokenizer
                .token_to_id(name)
                .ok_or_else(|| invalid(format!("Missing media token {name}")))?;
            if model[config_names[i]].as_u64() != Some(u64::from(ids[i])) {
                return Err(invalid(
                    "Tokenizer media IDs differ from model configuration",
                ));
            }
        }
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| invalid(e.to_string()))?;
        let budget = MediaBudget::new(config.memory_budget_bytes)?;
        let resolver = MediaResolver::new(
            config.allowed_image_hosts.clone(),
            budget.clone(),
            config.max_image_bytes,
            config.download_concurrency,
            config.timeout,
        )?;
        let mut frame_config = config.clone();
        frame_config.max_images = max_frames;
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            images: Gemma4ImageProcessor::from_config_with_budget(
                &processor["image_processor"],
                config.clone(),
                budget.clone(),
            )?,
            frames: Gemma4ImageProcessor::from_config_with_budget(
                video,
                frame_config,
                budget.clone(),
            )?,
            resolver,
            budget,
            workers: Arc::new(Semaphore::new(config.workers)),
            config,
            max_tokens,
            ids,
            prompts: prompts.unwrap_or_default(),
            default_prompt,
            video_fps,
            max_frames,
        })
    }

    pub async fn prepare(
        &self,
        messages: Vec<Message>,
        truncate: bool,
        direction: TruncationDirection,
        prompt_name: Option<String>,
    ) -> Result<PreparedMultimodal> {
        tokio::time::timeout(
            self.config.timeout,
            self.prepare_inner(messages, truncate, direction, prompt_name),
        )
        .await
        .map_err(|_| invalid("EmbeddingGemma2 media preprocessing timed out"))?
    }

    async fn prepare_inner(
        &self,
        mut messages: Vec<Message>,
        truncate: bool,
        direction: TruncationDirection,
        prompt_name: Option<String>,
    ) -> Result<PreparedMultimodal> {
        if messages.is_empty() {
            return Err(invalid("Messages cannot be empty"));
        }
        // The checkpoint's template puts system text first and concatenates the other turns.
        messages.sort_by_key(|m| !matches!(m.role, MessageRole::System));
        let mut text = String::new();
        let mut items = Vec::new();
        let mut memory = self.budget.reserve(0)?;
        let mut media_count = 0usize;
        for message in messages {
            match message.content {
                MessageContent::Text(content) => text.push_str(&content),
                MessageContent::Parts(parts) => {
                    let manual = parts
                        .iter()
                        .filter_map(|p| match p {
                            ContentPart::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .any(|t| {
                            ["<|image|>", "<|video|>", "<|audio|>"]
                                .iter()
                                .any(|s| t.contains(s))
                        });
                    for part in parts {
                        if matches!(&part, ContentPart::Text { .. }) {
                            if let ContentPart::Text { text: value } = part {
                                text.push_str(&value);
                            }
                            continue;
                        }
                        if matches!(message.role, MessageRole::System) {
                            return Err(invalid("System messages accept text only"));
                        }
                        media_count += 1;
                        if media_count > self.config.max_images {
                            return Err(invalid("Too many media inputs"));
                        }
                        match part {
                            ContentPart::ImageUrl { image_url } => {
                                if !matches!(image_url.detail, None | Some(ImageDetail::Auto)) {
                                    return Err(invalid("Image detail must be auto or omitted"));
                                }
                                let prepared = self.images.prepare(vec![image_url.url]).await?;
                                memory.merge(
                                    Arc::try_unwrap(prepared.memory).map_err(|_| {
                                        invalid("Image memory is unexpectedly shared")
                                    })?,
                                );
                                items.push(PreparedItem::Images(prepared.images, false));
                                if !manual {
                                    text.push_str("<|image|>");
                                }
                            }
                            ContentPart::VideoUrl { video_url } => {
                                let resolved = self.resolver.resolve(video_url.url).await?;
                                let frames = self.decode_video(&resolved.bytes).await?;
                                let mut patches = Vec::new();
                                // Resolve frames individually so 32 small PNGs do not
                                // each retain a worst-case encoded-byte reservation.
                                for (frame, _encoded_memory) in frames {
                                    let prepared = self.frames.prepare(vec![frame]).await?;
                                    memory.merge(Arc::try_unwrap(prepared.memory).map_err(
                                        |_| invalid("Image memory is unexpectedly shared"),
                                    )?);
                                    patches.extend(prepared.images);
                                }
                                items.push(PreparedItem::Images(patches, true));
                                if !manual {
                                    text.push_str("<|video|>");
                                }
                            }
                            ContentPart::InputAudio { input_audio } => {
                                let mime = match input_audio.format {
                                    AudioFormat::Wav => "audio/wav",
                                    AudioFormat::Mp3 => "audio/mpeg",
                                };
                                let resolved = self
                                    .resolver
                                    .resolve(format!("data:{mime};base64,{}", input_audio.data))
                                    .await?;
                                let decoder_format = match input_audio.format {
                                    AudioFormat::Wav => "wav",
                                    AudioFormat::Mp3 => "mp3",
                                };
                                let (samples, sample_memory) =
                                    self.decode_audio(&resolved.bytes, decoder_format).await?;
                                let (features, feature_memory) = prepare_audio_features(
                                    samples,
                                    sample_memory,
                                    self.budget.clone(),
                                    self.workers.clone(),
                                )
                                .await?;
                                memory.merge(feature_memory);
                                items.push(PreparedItem::Audio(Arc::new(features)));
                                if !manual {
                                    text.push_str("<|audio|>");
                                }
                            }
                            ContentPart::Text { .. } => unreachable!(),
                        }
                    }
                }
            }
        }
        // Prompt prefixes steer text tasks; media-only inputs do not receive a prefix.
        let has_text = text
            .replace("<|image|>", "")
            .replace("<|video|>", "")
            .replace("<|audio|>", "")
            .chars()
            .any(|c| !c.is_whitespace());
        if has_text {
            let prefix = match prompt_name {
                Some(name) => self
                    .prompts
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| invalid("Unknown prompt_name"))?,
                None => self.default_prompt.clone().unwrap_or_default(),
            };
            text.insert_str(0, &prefix);
        } else if prompt_name.is_some() {
            return Err(invalid("prompt_name requires text content"));
        }
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| invalid(e.to_string()))?;
        let mut input_ids = Vec::new();
        let mut images = Vec::new();
        let mut audios = Vec::new();
        let mut items = items.into_iter();
        for &id in encoding.get_ids() {
            if let Some(modality) = self.ids[..3].iter().position(|&m| m == id) {
                let item = items
                    .next()
                    .ok_or_else(|| invalid("Media placeholder has no matching input"))?;
                match (modality, item) {
                    (0, PreparedItem::Images(frames, false))
                    | (1, PreparedItem::Images(frames, true)) => {
                        for frame in frames {
                            input_ids.push(self.ids[3]);
                            images.push((input_ids.len(), frame.clone()));
                            input_ids.resize(input_ids.len() + frame.token_count(), id);
                            input_ids.push(self.ids[4]);
                        }
                    }
                    (2, PreparedItem::Audio(audio)) => {
                        input_ids.push(self.ids[5]);
                        audios.push((input_ids.len(), audio.clone()));
                        input_ids.resize(input_ids.len() + audio.token_count(), id);
                        input_ids.push(self.ids[6]);
                    }
                    _ => return Err(invalid("Media placeholder order differs from input order")),
                }
            } else {
                input_ids.push(id);
            }
        }
        if items.next().is_some() {
            return Err(invalid("Media input has no matching placeholder"));
        }
        if input_ids.is_empty() {
            return Err(invalid("Input cannot be empty"));
        }
        if input_ids.len() > self.max_tokens {
            if !truncate || !images.is_empty() || !audios.is_empty() {
                return Err(invalid(
                    "Input exceeds token limit; media spans cannot be truncated",
                ));
            }
            if direction == TruncationDirection::Left {
                input_ids.drain(..input_ids.len() - self.max_tokens);
            } else {
                input_ids.truncate(self.max_tokens);
            }
        }
        let positions = (0..input_ids.len() as u32).collect::<Vec<_>>();
        Ok(PreparedMultimodal {
            input_ids,
            media: Arc::new(MultimodalEncoding {
                images,
                audios,
                position_ids: std::array::from_fn(|_| positions.clone()),
                memory: Some(Arc::new(memory)),
            }),
        })
    }

    async fn decode_audio(
        &self,
        bytes: &[u8],
        format: &str,
    ) -> Result<(Vec<f32>, OwnedSemaphorePermit)> {
        let _worker = self
            .workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| invalid("Media workers closed"))?;
        let decode_memory = self.budget.reserve(480_000 * 8)?;
        let dir =
            tempfile::tempdir().map_err(|_| invalid("Media temporary directory unavailable"))?;
        let path = dir.path().join("audio");
        std::fs::write(&path, bytes).map_err(|_| invalid("Could not stage audio"))?;
        let mut child = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-f",
                format,
                "-threads",
                "1",
                "-i",
            ])
            .arg(&path)
            .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|_| invalid("Audio decoding requires ffmpeg"))?;
        let mut output = Vec::new();
        output.try_reserve_exact(480_000 * 4).map_err(|_| {
            TextEmbeddingsError::Overloaded(tokio::sync::TryAcquireError::NoPermits)
        })?;
        let mut stdout = child.stdout.take().unwrap();
        let mut chunk = [0u8; 8192];
        loop {
            let count = stdout
                .read(&mut chunk)
                .await
                .map_err(|_| invalid("Audio decoding failed"))?;
            if count == 0 {
                break;
            }
            if count > 480_000 * 4 - output.len() {
                return Err(invalid("Audio exceeds the 30 second limit"));
            }
            output.extend_from_slice(&chunk[..count]);
        }
        if !child
            .wait()
            .await
            .map_err(|_| invalid("Audio decoder failed"))?
            .success()
            || output.len() % 4 != 0
        {
            return Err(invalid("Invalid or unsupported audio"));
        }
        let mut samples = Vec::new();
        samples.try_reserve_exact(output.len() / 4).map_err(|_| {
            TextEmbeddingsError::Overloaded(tokio::sync::TryAcquireError::NoPermits)
        })?;
        samples.extend(
            output
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
        );
        if samples.len() < 321 || samples.iter().any(|v| !v.is_finite()) {
            return Err(invalid("Audio is too short or contains invalid samples"));
        }
        // Keep the reservation with the returned samples through extraction.
        Ok((samples, decode_memory))
    }

    async fn decode_video(&self, bytes: &[u8]) -> Result<Vec<(String, OwnedSemaphorePermit)>> {
        let format = if bytes.get(4..8) == Some(b"ftyp".as_slice()) {
            "mov"
        } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
            "matroska"
        } else {
            return Err(invalid("Video must be MP4 or WebM"));
        };
        let _worker = self
            .workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| invalid("Media workers closed"))?;
        let dir =
            tempfile::tempdir().map_err(|_| invalid("Media temporary directory unavailable"))?;
        let path = dir.path().join("video");
        std::fs::write(&path, bytes).map_err(|_| invalid("Could not stage video"))?;
        let metadata = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-f",
                format,
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
                "-of",
                "json",
            ])
            .arg(&path)
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|_| invalid("Video decoding requires ffmpeg and ffprobe"))?;
        if !metadata.status.success() {
            return Err(invalid("Invalid or unsupported video"));
        }
        let metadata: serde_json::Value = serde_json::from_slice(&metadata.stdout)
            .map_err(|_| invalid("Invalid video metadata"))?;
        let stream = &metadata["streams"][0];
        let width = stream["width"].as_u64().unwrap_or(0);
        let height = stream["height"].as_u64().unwrap_or(0);
        if width == 0
            || height == 0
            || width.saturating_mul(height) > self.config.max_decoded_pixels
        {
            return Err(invalid("Video exceeds decoded pixel limit"));
        }
        let _decoded_memory = self.budget.reserve((width * height * 8) as usize)?;
        let duration = stream["duration"]
            .as_str()
            .or(metadata["format"]["duration"].as_str())
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0 && *v <= 600.0)
            .ok_or_else(|| invalid("Video must have a valid duration of at most 10 minutes"))?;
        let fps = stream["avg_frame_rate"]
            .as_str()
            .and_then(|s| s.split_once('/'))
            .and_then(|(a, b)| Some(a.parse::<f64>().ok()? / b.parse::<f64>().ok()?))
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| invalid("Invalid video frame rate"))?;
        let total = stream["nb_frames"]
            .as_str()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or((duration * fps).round() as usize);
        let indices = sample_video_frames(total, fps, duration, self.video_fps, self.max_frames);
        let mut unique_indices = indices.clone();
        unique_indices.dedup();
        let select = unique_indices
            .iter()
            .map(|n| format!("eq(n\\,{n})"))
            .collect::<Vec<_>>()
            .join("+");
        let status = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-f",
                format,
                "-threads",
                "1",
                "-i",
            ])
            .arg(&path)
            .args([
                "-an",
                "-vf",
                &format!("select={select}"),
                "-vsync",
                "0",
                "-frames:v",
                &unique_indices.len().to_string(),
                "-threads",
                "1",
            ])
            .arg(dir.path().join("frame-%03d.png"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .status()
            .await
            .map_err(|_| invalid("Video decoding failed"))?;
        if !status.success() {
            return Err(invalid("Video decoding failed"));
        }
        let mut sources = Vec::new();
        let mut source_memory = Vec::new();
        for i in 1..=unique_indices.len() {
            let (source, reservation) = read_video_frame(
                &dir.path().join(format!("frame-{i:03}.png")),
                self.config.max_image_bytes,
                &self.budget,
            )?;
            source_memory.push(reservation);
            sources.push(source);
        }
        indices
            .iter()
            .map(|index| {
                let source = &sources[unique_indices.binary_search(index).unwrap()];
                let memory = self.budget.reserve(source.len())?;
                let mut copy = String::new();
                copy.try_reserve_exact(source.len()).map_err(|_| {
                    TextEmbeddingsError::Overloaded(tokio::sync::TryAcquireError::NoPermits)
                })?;
                copy.push_str(source);
                Ok((copy, memory))
            })
            .collect()
    }
}

fn read_video_frame(
    path: &Path,
    max_bytes: usize,
    budget: &MediaBudget,
) -> Result<(String, OwnedSemaphorePermit)> {
    use std::io::Read;
    const PREFIX: &str = "data:image/png;base64,";
    let mut file = std::fs::File::open(path).map_err(|_| invalid("Missing decoded video frame"))?;
    let length = usize::try_from(
        file.metadata()
            .map_err(|_| invalid("Missing decoded video frame"))?
            .len(),
    )
    .map_err(|_| invalid("Decoded video frame exceeds byte limit"))?;
    if length > max_bytes {
        return Err(invalid("Decoded video frame exceeds byte limit"));
    }
    let encoded = base64::encoded_len(length, true)
        .and_then(|size| size.checked_add(PREFIX.len()))
        .ok_or_else(|| invalid("Decoded video frame allocation overflow"))?;
    let bytes = length
        .checked_add(encoded)
        .ok_or_else(|| invalid("Decoded video frame allocation overflow"))?;
    // Charge simultaneous compressed frame and final data URL before reading.
    let memory = budget.reserve(bytes)?;
    let mut frame = Vec::new();
    frame
        .try_reserve_exact(length)
        .map_err(|_| TextEmbeddingsError::Overloaded(tokio::sync::TryAcquireError::NoPermits))?;
    frame.resize(length, 0);
    file.read_exact(&mut frame)
        .map_err(|_| invalid("Missing decoded video frame"))?;
    let mut extra = [0u8; 1];
    if file
        .read(&mut extra)
        .map_err(|_| invalid("Missing decoded video frame"))?
        != 0
    {
        return Err(invalid("Decoded video frame changed during read"));
    }
    let mut source = String::new();
    source
        .try_reserve_exact(encoded)
        .map_err(|_| TextEmbeddingsError::Overloaded(tokio::sync::TryAcquireError::NoPermits))?;
    source.push_str(PREFIX);
    STANDARD.encode_string(&frame, &mut source);
    Ok((source, memory))
}

fn sample_video_frames(
    total: usize,
    fps: f64,
    duration: f64,
    target_fps: f64,
    max_frames: usize,
) -> Vec<usize> {
    let count = ((duration * target_fps) as usize).max(1);
    let selected = count.min(max_frames);
    (0..selected)
        .map(|i| {
            let index = if count > max_frames && selected > 1 {
                i * (count - 1) / (selected - 1)
            } else {
                i
            };
            ((index as f64 * fps / target_fps) as usize).min(total.saturating_sub(1))
        })
        .collect()
}

async fn prepare_audio_features(
    samples: Vec<f32>,
    sample_memory: OwnedSemaphorePermit,
    budget: MediaBudget,
    workers: Arc<Semaphore>,
) -> Result<(AudioFeatures, OwnedSemaphorePermit)> {
    let worker = workers
        .acquire_owned()
        .await
        .map_err(|_| invalid("Media workers closed"))?;
    // Reserve F32 features, their native conversion copy, and mask before allocation.
    let frames = audio_feature_frames(samples.len())?;
    let feature_memory = budget.reserve(frames * (128 * 8 + 1))?;
    tokio::task::spawn_blocking(move || {
        // Cancellation cannot release admission while blocking work owns samples.
        let _worker = worker;
        let _sample_memory = sample_memory;
        let features = extract_audio_features(&samples)?;
        Ok((features, feature_memory))
    })
    .await
    .map_err(|_| invalid("Audio feature extraction failed"))?
}

fn audio_feature_frames(samples: usize) -> Result<usize> {
    if !(321..=480_000).contains(&samples) {
        return Err(invalid("Invalid audio sample count"));
    }
    // The sample limit makes padding, frame counts and output byte counts safe.
    Ok((samples.div_ceil(128) * 128 + 160 - 321) / 160 + 1)
}

/// Gemma4's magnitude (not power) log-mel spectrogram with semicausal Hann windows.
pub(crate) fn extract_audio_features(samples: &[f32]) -> Result<AudioFeatures> {
    use rustfft::{num_complex::Complex, FftPlanner};
    const FRAME: usize = 320;
    const HOP: usize = 160;
    const FFT: usize = 512;
    const BINS: usize = 128;
    let frames = audio_feature_frames(samples.len())?;
    let window: Vec<f32> = (0..FRAME)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / FRAME as f64).cos()) as f32)
        .collect();
    let mel_max = 2595.0 * (1.0f64 + 8000.0 / 700.0).log10();
    let edges: Vec<f64> = (0..BINS + 2)
        .map(|i| 700.0 * (10.0f64.powf(mel_max * i as f64 / (BINS + 1) as f64 / 2595.0) - 1.0))
        .collect();
    let filters: Vec<Vec<f64>> = (0..BINS)
        .map(|m| {
            (0..=FFT / 2)
                .map(|k| {
                    let freq = k as f64 * 16000.0 / FFT as f64;
                    ((freq - edges[m]) / (edges[m + 1] - edges[m]))
                        .min((edges[m + 2] - freq) / (edges[m + 2] - edges[m + 1]))
                        .max(0.0)
                })
                .collect()
        })
        .collect();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(FFT);
    let mut values = Vec::new();
    values
        .try_reserve_exact(frames * BINS)
        .map_err(|_| TextEmbeddingsError::Overloaded(tokio::sync::TryAcquireError::NoPermits))?;
    let mut mask = Vec::new();
    mask.try_reserve_exact(frames)
        .map_err(|_| TextEmbeddingsError::Overloaded(tokio::sync::TryAcquireError::NoPermits))?;
    // Explicit scratch avoids an infallible allocation on every FFT frame.
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(fft.get_inplace_scratch_len())
        .map_err(|_| TextEmbeddingsError::Overloaded(tokio::sync::TryAcquireError::NoPermits))?;
    scratch.resize(fft.get_inplace_scratch_len(), Complex::default());
    let mut buffer = [Complex::default(); FFT];
    let mut magnitude = [0.0; FFT / 2 + 1];
    for frame in 0..frames {
        let valid = frame * HOP + FRAME < samples.len() + FRAME / 2;
        mask.push(u8::from(valid));
        buffer.fill(Complex::default());
        for i in 0..FRAME {
            let position = frame * HOP + i;
            if let Some(position) = position.checked_sub(FRAME / 2) {
                if let Some(&sample) = samples.get(position) {
                    buffer[i].re = (sample * window[i]) as f64;
                }
            }
        }
        fft.process_with_scratch(&mut buffer, &mut scratch);
        for (value, complex) in magnitude.iter_mut().zip(&buffer) {
            *value = complex.norm();
        }
        for filter in &filters {
            let mel = magnitude
                .iter()
                .zip(filter)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            values.push(if valid {
                (mel + 0.001).ln() as f32
            } else {
                0.0
            });
        }
    }
    Ok(AudioFeatures {
        values,
        mask,
        feature_size: BINS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancelled_audio_extraction_retains_admission_until_blocking_work_finishes() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let budget = MediaBudget::new(8 * 1024 * 1024).unwrap();
            let workers = Arc::new(Semaphore::new(1));
            let (release, blocked) = std::sync::mpsc::channel();
            let (started, ready) = tokio::sync::oneshot::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                blocked
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            });
            ready.await.unwrap();
            let samples = vec![0.0; 16000];
            let sample_memory = budget.reserve(samples.len() * 4).unwrap();
            let request = tokio::spawn(prepare_audio_features(
                samples,
                sample_memory,
                budget.clone(),
                workers.clone(),
            ));
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while workers.available_permits() != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            request.abort();
            assert!(request.await.unwrap_err().is_cancelled());
            assert_eq!(workers.available_permits(), 0);
            assert!(budget.reserve(8 * 1024 * 1024).is_err());
            release.send(()).unwrap();
            blocker.await.unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while workers.available_permits() == 0 || budget.reserve(8 * 1024 * 1024).is_err() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert!(budget.reserve(8 * 1024 * 1024).is_ok());
        });
    }

    #[test]
    fn video_uniform_sampling_matches_processor() {
        assert_eq!(
            sample_video_frames(300, 30.0, 10.0, 1.0, 32),
            (0..10).map(|i| i * 30).collect::<Vec<_>>()
        );
        let frames = sample_video_frames(3600, 30.0, 120.0, 1.0, 32);
        assert_eq!(frames.len(), 32);
        assert_eq!(frames[0], 0);
        assert_eq!(frames[31], 3570);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frame.png");
        std::fs::write(&path, [1, 2, 3]).unwrap();
        let budget = MediaBudget::new(1024).unwrap();
        assert!(read_video_frame(&path, 2, &budget).is_err());
        let held = budget.reserve(1024).unwrap();
        assert!(matches!(
            read_video_frame(&path, 3, &budget),
            Err(TextEmbeddingsError::Overloaded(_))
        ));
        drop(held);
        let (source, memory) = read_video_frame(&path, 3, &budget).unwrap();
        assert_eq!(source, "data:image/png;base64,AQID");
        assert!(budget.reserve(1024).is_err());
        drop(memory);
        assert!(budget.reserve(1024).is_ok());
    }
    #[test]
    fn audio_frame_and_token_counts_follow_two_stride_two_convolutions() {
        let features = extract_audio_features(&vec![0.0; 16000]).unwrap();
        assert_eq!(features.mask.len(), 99);
        assert!(audio_feature_frames(320).is_err());
        assert!(audio_feature_frames(480_001).is_err());
        assert!(audio_feature_frames(usize::MAX).is_err());
        assert_eq!(features.token_count(), 25);
        assert!(features
            .values
            .iter()
            .all(|v| (*v - 0.001f32.ln()).abs() < 1e-6));
    }
}
