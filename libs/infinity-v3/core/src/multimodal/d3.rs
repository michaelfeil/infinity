//! D3's bounded image preparation and timestamped, two-frame video patches.
use super::{
    image::QwenImageProcessor,
    media::{MediaBudget, MediaResolver},
    MultimodalConfig,
};
use crate::TextEmbeddingsError;
use infinity_v3_backend::ImagePatches;
use std::{io::Write, path::Path, process::Stdio, sync::Arc};
use tokenizers::Tokenizer;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

type Result<T> = std::result::Result<T, TextEmbeddingsError>;
fn invalid(message: impl Into<String>) -> TextEmbeddingsError {
    TextEmbeddingsError::Validation(message.into())
}

#[derive(Clone)]
pub struct D3MediaProcessor {
    images: QwenImageProcessor,
    resolver: MediaResolver,
    budget: MediaBudget,
    workers: Arc<Semaphore>,
    config: MultimodalConfig,
}

pub struct PreparedD3Media {
    images: Vec<Arc<ImagePatches>>,
    videos: Vec<Vec<(f64, Arc<ImagePatches>)>>,
    pub memory: Arc<OwnedSemaphorePermit>,
}

type Spans = Vec<(usize, Arc<ImagePatches>)>;
impl PreparedD3Media {
    pub fn encode_prefix(
        &self,
        tokenizer: &Tokenizer,
    ) -> std::result::Result<(Vec<u32>, Spans), String> {
        let token = |s| {
            tokenizer
                .token_to_id(s)
                .ok_or_else(|| format!("Missing D3 media token {s}"))
        };
        let start = token("<|vision_start|>")?;
        let end = token("<|vision_end|>")?;
        let image_id = token("<|image_pad|>")?;
        let video_id = token("<|video_pad|>")?;
        let mut ids = Vec::new();
        let mut spans = Vec::new();
        let append = |ids: &mut Vec<u32>, spans: &mut Spans, patch: &Arc<ImagePatches>, pad| {
            ids.push(start);
            spans.push((ids.len(), patch.clone()));
            ids.resize(ids.len() + patch.token_count(), pad);
            ids.push(end);
        };
        for image in &self.images {
            append(&mut ids, &mut spans, image, image_id);
        }
        for video in &self.videos {
            // The native chat template wraps the video placeholder. Replacing that
            // placeholder adds timestamped frame groups inside the original wrapper.
            ids.push(start);
            for (timestamp, patches) in video {
                let stamp = format!("<{timestamp:.1} seconds>");
                ids.extend_from_slice(
                    tokenizer
                        .encode(stamp.as_str(), false)
                        .map_err(|e| e.to_string())?
                        .get_ids(),
                );
                append(&mut ids, &mut spans, patches, video_id);
            }
            ids.push(end);
        }
        Ok((ids, spans))
    }
}

impl D3MediaProcessor {
    pub fn load(root: &Path, config: MultimodalConfig) -> Result<Self> {
        if config.workers == 0 || config.max_images == 0 || config.memory_budget_bytes == 0 {
            return Err(invalid("Invalid D3 media limits"));
        }
        let budget = MediaBudget::new(config.memory_budget_bytes)?;
        let resolver = MediaResolver::new(
            config.allowed_image_hosts.clone(),
            budget.clone(),
            config.max_image_bytes,
            config.download_concurrency,
            config.timeout,
        )?;
        Ok(Self {
            images: QwenImageProcessor::load_d3(root).map_err(invalid)?,
            resolver,
            workers: Arc::new(Semaphore::new(config.workers)),
            budget,
            config,
        })
    }

    pub async fn prepare(
        &self,
        images: Vec<String>,
        videos: Vec<String>,
    ) -> Result<PreparedD3Media> {
        if images.len() + videos.len() == 0 || images.len() + videos.len() > self.config.max_images
        {
            return Err(invalid(format!(
                "D3 accepts at most {} media sources per request",
                self.config.max_images
            )));
        }
        tokio::time::timeout(self.config.timeout, self.prepare_inner(images, videos))
            .await
            .map_err(|_| invalid("D3 media preprocessing timed out"))?
    }

    async fn prepare_inner(
        &self,
        images: Vec<String>,
        videos: Vec<String>,
    ) -> Result<PreparedD3Media> {
        // Locals drop in reverse order: buffers must be freed before their permits.
        let mut held = Vec::new();
        let mut prepared_images = Vec::new();
        let mut prepared_videos = Vec::new();
        for source in images {
            let resolved = self.resolver.resolve(source).await?;
            let worker = self
                .workers
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| invalid("Media workers closed"))?;
            let processor = self.images.clone();
            let budget = self.budget.clone();
            let max_pixels = self.config.max_decoded_pixels;
            let (image, memory) = tokio::task::spawn_blocking(move || {
                let _worker = worker;
                let plan = processor
                    .inspect(&resolved.bytes, max_pixels)
                    .map_err(invalid)?;
                let memory = budget.reserve(plan.memory_bytes)?;
                let image = processor
                    .prepare(&resolved.bytes, &plan, max_pixels)
                    .map_err(invalid)?;
                Ok::<_, TextEmbeddingsError>((Arc::new(image), memory))
            })
            .await
            .map_err(|_| invalid("D3 image worker failed"))??;
            prepared_images.push(image);
            held.push(memory);
        }
        let mut video_tokens = 0;
        for source in videos {
            let resolved = self.resolver.resolve(source).await?;
            let worker = self
                .workers
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| invalid("Media workers closed"))?;
            let (directory, indices, fps, spool_memory) =
                self.decode_video(&resolved.bytes).await?;
            drop(resolved);
            let processor = self.images.video_frames(indices.len() + indices.len() % 2);
            let budget = self.budget.clone();
            let config = self.config.clone();
            let (frames, memory) = tokio::task::spawn_blocking(move || {
                let _worker = worker;
                // Keep all encoded spool reservations until the files are consumed.
                let _spool_memory = spool_memory;
                // Preserve that order on worker errors as well as normal returns.
                let mut held = Vec::new();
                let mut groups = Vec::new();
                for pair in indices.chunks(2) {
                    let mut patches = Vec::new();
                    for index in pair {
                        let path = directory.path().join(format!("frame-{index:08}.png"));
                        let len = std::fs::metadata(&path)
                            .map_err(|_| invalid("Missing video frame"))?
                            .len() as usize;
                        if len > config.max_image_bytes {
                            return Err(invalid("Decoded video frame exceeds byte limit"));
                        }
                        let _encoded = budget.reserve(len)?;
                        let bytes = std::fs::read(&path)
                            .map_err(|_| invalid("Could not read video frame"))?;
                        let plan = processor
                            .inspect(&bytes, config.max_decoded_pixels)
                            .map_err(invalid)?;
                        let memory = budget.reserve(plan.memory_bytes)?;
                        patches.push(
                            processor
                                .prepare(&bytes, &plan, config.max_decoded_pixels)
                                .map_err(invalid)?,
                        );
                        held.push(memory);
                    }
                    if patches.len() == 1 {
                        patches.push(patches[0].clone());
                    }
                    let second = patches.pop().unwrap();
                    let mut first = patches.pop().unwrap();
                    pair_patches(&mut first, &second).map_err(invalid)?;
                    let last = *pair.last().unwrap();
                    groups.push((
                        (pair[0] as f64 + last as f64) / (2.0 * fps),
                        Arc::new(first),
                    ));
                }
                Ok::<_, TextEmbeddingsError>((groups, held))
            })
            .await
            .map_err(|_| invalid("D3 video worker failed"))??;
            held.extend(memory);
            video_tokens += frames.iter().map(|(_, p)| p.token_count()).sum::<usize>();
            if video_tokens > 16_384 {
                return Err(invalid("D3 videos exceed 16384 vision tokens"));
            }
            prepared_videos.push(frames);
        }
        let bytes = prepared_images
            .iter()
            .chain(prepared_videos.iter().flatten().map(|(_, p)| p))
            .try_fold(0usize, |total, p| {
                total.checked_add(p.pixels.len().checked_mul(8)?)
            })
            .ok_or_else(|| invalid("D3 media allocation is too large"))?;
        // Every allocation was reserved before decoding. Consolidate the output reservation
        // so the shared queue can retain it until the final question completes.
        let memory = Arc::new(retain_media_reservation(&mut held, bytes, &self.budget)?);
        Ok(PreparedD3Media {
            images: prepared_images,
            videos: prepared_videos,
            memory,
        })
    }

    async fn decode_video(
        &self,
        bytes: &[u8],
    ) -> Result<(
        tempfile::TempDir,
        Vec<usize>,
        f64,
        Vec<OwnedSemaphorePermit>,
    )> {
        let format = if bytes.get(4..8) == Some(b"ftyp".as_slice()) {
            "mov"
        } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
            "matroska"
        } else {
            return Err(invalid("D3 video must be MP4, MOV, WebM or MKV"));
        };
        if bytes.len() > 32_000_000 {
            return Err(invalid("D3 video exceeds 32 MB"));
        }
        let directory =
            tempfile::tempdir().map_err(|_| invalid("Could not create media directory"))?;
        let path = directory.path().join("video");
        std::fs::write(&path, bytes).map_err(|_| invalid("Could not stage video"))?;
        let output = Command::new("ffprobe")
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
            .map_err(|_| invalid("D3 video requires ffmpeg and ffprobe"))?;
        if !output.status.success() {
            return Err(invalid("Invalid video"));
        }
        let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|_| invalid("Invalid video metadata"))?;
        let stream = &metadata["streams"][0];
        let width = stream["width"].as_u64().unwrap_or(0);
        let height = stream["height"].as_u64().unwrap_or(0);
        if width == 0
            || height == 0
            || width.saturating_mul(height) > self.config.max_decoded_pixels.min(8_294_400)
        {
            return Err(invalid("D3 video exceeds decoded pixel limit"));
        }
        let _decode = self.budget.reserve((width * height * 8) as usize)?;
        let fps = stream["avg_frame_rate"]
            .as_str()
            .and_then(|s| s.split_once('/'))
            .and_then(|(a, b)| Some(a.parse::<f64>().ok()? / b.parse::<f64>().ok()?))
            .filter(|fps| fps.is_finite() && (0.1..=1000.).contains(fps))
            .unwrap_or(24.);
        let _duration = stream["duration"]
            .as_str()
            .or(metadata["format"]["duration"].as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0. && *v <= 300.)
            .ok_or_else(|| invalid("D3 video requires a valid duration of at most 300 seconds"))?;
        let total = match stream["nb_frames"]
            .as_str()
            .and_then(|v| v.parse::<usize>().ok())
        {
            Some(total) => total,
            None => {
                // Container duration and average FPS do not determine a VFR frame count.
                // Decode only after validating geometry and reserving decoder memory above;
                // the enclosing preparation timeout and kill_on_drop also cover this pass.
                let output = Command::new("ffprobe")
                    .args([
                        "-v",
                        "error",
                        "-protocol_whitelist",
                        "file,pipe",
                        "-f",
                        format,
                        "-threads",
                        "1",
                        "-select_streams",
                        "v:0",
                        "-count_frames",
                        "-show_entries",
                        "stream=nb_read_frames",
                        "-of",
                        "json",
                    ])
                    .arg(&path)
                    .stderr(Stdio::null())
                    .kill_on_drop(true)
                    .output()
                    .await
                    .map_err(|_| invalid("D3 video frame counting requires ffprobe"))?;
                if !output.status.success() {
                    return Err(invalid("Video frame counting failed"));
                }
                let counted: serde_json::Value = serde_json::from_slice(&output.stdout)
                    .map_err(|_| invalid("Invalid video frame count"))?;
                counted["streams"][0]["nb_read_frames"]
                    .as_str()
                    .and_then(|v| v.parse::<usize>().ok())
                    .ok_or_else(|| invalid("Video frame count is unavailable"))?
            }
        };
        if total == 0 || total as f64 / fps > 300. {
            return Err(invalid("Video contains no frames"));
        }
        let indices = sample_indices(total, fps);
        let select = indices
            .iter()
            .map(|n| format!("eq(n\\,{n})"))
            .collect::<Vec<_>>()
            .join("+");
        // Stream unchanged PNGs so byte limits are enforced before writing their payloads.
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
            .args([
                "-an",
                "-vf",
                &format!("select={select}"),
                "-vsync",
                "0",
                "-threads",
                "1",
                "-frames:v",
                &indices.len().to_string(),
            ])
            .args(["-f", "image2pipe", "-c:v", "png", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|_| invalid("D3 video decoding requires ffmpeg"))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| invalid("Missing video output"))?;
        let mut spool_memory = Vec::new();
        let decoded = async {
            for index in &indices {
                let frame_path = directory.path().join(format!("frame-{index:08}.png"));
                let mut frame = std::fs::File::create(frame_path)
                    .map_err(|_| invalid("Could not stage video frame"))?;
                spool_memory.extend(
                    copy_bounded_png(
                        &mut stdout,
                        &mut frame,
                        self.config.max_image_bytes,
                        &self.budget,
                    )
                    .await?,
                );
            }
            let mut trailing = [0];
            if stdout
                .read(&mut trailing)
                .await
                .map_err(|_| invalid("Could not read video output"))?
                != 0
            {
                return Err(invalid("Unexpected extra video frames"));
            }
            Ok::<_, TextEmbeddingsError>(())
        }
        .await;
        if let Err(error) = decoded {
            // Reap the decoder on explicit failures; kill_on_drop also covers cancellation.
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(error);
        }
        if !child
            .wait()
            .await
            .map_err(|_| invalid("Video decoding failed"))?
            .success()
        {
            return Err(invalid("Video decoding failed"));
        }
        Ok((directory, indices, fps, spool_memory))
    }
}

// Merge permits without exposing live pixel allocations to a release/reacquire race.
// Decoder/resize scratch is gone; only release units beyond output plus its upload copy.
fn retain_media_reservation(
    held: &mut Vec<OwnedSemaphorePermit>,
    bytes: usize,
    budget: &MediaBudget,
) -> Result<OwnedSemaphorePermit> {
    if held.is_empty() {
        return Err(invalid("Missing D3 media reservation"));
    }
    let required = bytes.div_ceil(1024);
    let reserved = held
        .iter()
        .map(OwnedSemaphorePermit::num_permits)
        .sum::<usize>();
    if required > reserved {
        // Keep reservations with the caller's buffers if acquiring the deficit fails.
        held.push(budget.reserve((required - reserved) * 1024)?);
    }
    let mut memory = held
        .pop()
        .ok_or_else(|| invalid("Missing D3 media reservation"))?;
    for permit in held.drain(..) {
        memory.merge(permit);
    }
    let excess = memory.num_permits() - required;
    if excess > 0 {
        drop(
            memory
                .split(excess)
                .ok_or_else(|| invalid("Invalid D3 media reservation"))?,
        );
    }
    Ok(memory)
}

/// PNG chunk lengths let us reject oversized output before reading or writing its payload.
async fn copy_bounded_png<R: AsyncRead + Unpin, W: Write>(
    reader: &mut R,
    writer: &mut W,
    max_bytes: usize,
    budget: &MediaBudget,
) -> Result<Vec<OwnedSemaphorePermit>> {
    let mut signature = [0; 8];
    reader
        .read_exact(&mut signature)
        .await
        .map_err(|_| invalid("Missing decoded video frame"))?;
    if signature != *b"\x89PNG\r\n\x1a\n" || max_bytes < signature.len() {
        return Err(invalid("Invalid decoded video frame"));
    }
    let mut held = vec![budget.reserve(signature.len())?];
    writer
        .write_all(&signature)
        .map_err(|_| invalid("Could not stage video frame"))?;
    let mut size = signature.len();
    let mut buffer = [0; 65_536];
    loop {
        let mut header = [0; 8];
        reader
            .read_exact(&mut header)
            .await
            .map_err(|_| invalid("Truncated decoded video frame"))?;
        let payload = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
        let next = size
            .checked_add(payload)
            .and_then(|n| n.checked_add(12))
            .filter(|n| *n <= max_bytes)
            .ok_or_else(|| invalid("Decoded video frame exceeds byte limit"))?;
        let extra_units = next.div_ceil(1024) - size.div_ceil(1024);
        if extra_units > 0 {
            held.push(budget.reserve(extra_units * 1024)?);
        }
        writer
            .write_all(&header)
            .map_err(|_| invalid("Could not stage video frame"))?;
        let mut remaining = payload + 4; // Include the chunk CRC without a payload-sized allocation.
        while remaining > 0 {
            let count = remaining.min(buffer.len());
            reader
                .read_exact(&mut buffer[..count])
                .await
                .map_err(|_| invalid("Truncated decoded video frame"))?;
            writer
                .write_all(&buffer[..count])
                .map_err(|_| invalid("Could not stage video frame"))?;
            remaining -= count;
        }
        size = next;
        if &header[4..] == b"IEND" {
            if payload != 0 {
                return Err(invalid("Invalid decoded video frame"));
            }
            return Ok(held);
        }
    }
}

fn sample_indices(total: usize, fps: f64) -> Vec<usize> {
    let count = ((total as f64 / fps * 2.) as usize).clamp(4, 32).min(total);
    (0..count)
        .map(|i| {
            if count == 1 {
                0
            } else {
                (i as f64 * (total - 1) as f64 / (count - 1) as f64).round_ties_even() as usize
            }
        })
        .collect()
}

fn pair_patches(
    first: &mut ImagePatches,
    second: &ImagePatches,
) -> std::result::Result<(), String> {
    if first.grid_thw != second.grid_thw
        || first.patch_dim != second.patch_dim
        || first.pixels.len() != second.pixels.len()
    {
        return Err("Video frames have different patch geometry".into());
    }
    // Image patches repeat one frame at both temporal positions. Replace the second
    // temporal position for each channel of every patch with the next frame.
    let channel = first.patch_dim / 3;
    let frame = channel / 2;
    for (a, b) in first
        .pixels
        .chunks_exact_mut(channel)
        .zip(second.pixels.chunks_exact(channel))
    {
        a[frame..].copy_from_slice(&b[..frame]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    #[ignore = "requires ffmpeg and ffprobe"]
    async fn video_sampling_counts_frames_when_container_metadata_has_none() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("preprocessor_config.json"),
            r#"{"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#,
        )
        .unwrap();
        let path = root.path().join("variable.mkv");
        assert!(Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=32x32:rate=2:duration=2",
                "-vf",
                "setpts=2*PTS",
                "-vsync",
                "0",
                "-threads",
                "1",
                "-c:v",
                "ffv1",
            ])
            .arg(&path)
            .status()
            .await
            .unwrap()
            .success());
        // Four actual frames, but duration (3.5 seconds) * nominal FPS (2) estimates seven.
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=nb_frames,avg_frame_rate:format=duration",
                "-of",
                "json",
            ])
            .arg(&path)
            .output()
            .await
            .unwrap();
        assert!(output.status.success());
        let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(metadata["streams"][0]["nb_frames"].is_null());
        assert_eq!(metadata["streams"][0]["avg_frame_rate"], "2/1");
        assert_eq!(metadata["format"]["duration"], "3.500000");
        let processor = D3MediaProcessor::load(root.path(), MultimodalConfig::default()).unwrap();
        let (frames, indices, fps, reservations) = processor
            .decode_video(&std::fs::read(path).unwrap())
            .await
            .unwrap();
        assert_eq!(fps, 2.);
        assert_eq!(indices, vec![0, 1, 2, 3]);
        for index in indices {
            let frame = frames.path().join(format!("frame-{index:08}.png"));
            assert_eq!(image::open(frame).unwrap().to_rgb8().dimensions(), (32, 32));
        }
        assert!(!reservations.is_empty());
    }

    #[test]
    fn media_transfer_retains_output_and_releases_only_scratch() {
        let budget = MediaBudget::new(4096).unwrap();
        let mut held = vec![budget.reserve(2048).unwrap(), budget.reserve(2048).unwrap()];
        let output = retain_media_reservation(&mut held, 3072, &budget).unwrap();
        assert_eq!(output.num_permits(), 3);
        assert!(budget.reserve(2048).is_err());
        let competitor = budget.reserve(1024).unwrap();
        assert!(budget.reserve(1024).is_err());
        drop(output);
        assert!(budget.reserve(3072).is_ok());
        drop(competitor);
        let mut held = vec![budget.reserve(2048).unwrap()];
        let output = retain_media_reservation(&mut held, 3072, &budget).unwrap();
        assert_eq!(output.num_permits(), 3);
        assert!(budget.reserve(2048).is_err());
        // A competing request cannot steal the three units that remain live during transfers.
        use std::sync::atomic::{AtomicBool, Ordering};
        let running = Arc::new(AtomicBool::new(true));
        let competing_budget = budget.clone();
        let competing_running = running.clone();
        let competitor = std::thread::spawn(move || {
            while competing_running.load(Ordering::Relaxed) {
                assert!(competing_budget.reserve(2048).is_err());
                std::thread::yield_now();
            }
        });
        let mut output = output;
        for _ in 0..1000 {
            let scratch = budget.reserve(1024).unwrap();
            output = retain_media_reservation(&mut vec![output, scratch], 3072, &budget).unwrap();
        }
        running.store(false, Ordering::Relaxed);
        competitor.join().unwrap();
        drop(output);
        let mut held = vec![budget.reserve(3072).unwrap()];
        assert!(retain_media_reservation(&mut held, 5120, &budget).is_err());
        assert_eq!(held[0].num_permits(), 3);
        assert!(budget.reserve(2048).is_err());
    }

    #[tokio::test]
    async fn png_stream_preserves_full_pixels_and_bounds_each_frame() {
        let pixels = image::RgbImage::from_fn(31, 29, |x, y| {
            image::Rgb([(x * 7) as u8, (y * 11) as u8, (x ^ y) as u8])
        });
        let mut png = std::io::Cursor::new(Vec::new());
        pixels.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let png = png.into_inner();
        let mut output = Vec::new();
        let budget = MediaBudget::new(png.len().div_ceil(1024) * 1024).unwrap();
        let held = copy_bounded_png(&mut png.as_slice(), &mut output, png.len(), &budget)
            .await
            .unwrap();
        assert_eq!(output, png);
        assert_eq!(image::load_from_memory(&output).unwrap().to_rgb8(), pixels);
        // The spool remains reserved until consumed, including across separate frames.
        assert!(
            copy_bounded_png(&mut png.as_slice(), &mut Vec::new(), png.len(), &budget)
                .await
                .is_err()
        );
        drop(held);
        assert!(
            copy_bounded_png(&mut png.as_slice(), &mut Vec::new(), png.len(), &budget)
                .await
                .is_ok()
        );
        let mut oversized = b"\x89PNG\r\n\x1a\n".to_vec();
        oversized.extend_from_slice(&u32::MAX.to_be_bytes());
        oversized.extend_from_slice(b"IDAT");
        let mut written = Vec::new();
        assert!(copy_bounded_png(
            &mut oversized.as_slice(),
            &mut written,
            1024,
            &MediaBudget::new(1024).unwrap()
        )
        .await
        .is_err());
        assert_eq!(written, b"\x89PNG\r\n\x1a\n"); // No payload is read, allocated, or written.
    }

    #[test]
    fn sampling_matches_d3_linspace_and_ties_even() {
        assert_eq!(sample_indices(1, 24.), vec![0]);
        assert_eq!(sample_indices(30, 30.), vec![0, 10, 19, 29]);
        assert_eq!(
            sample_indices(121, 30.),
            vec![0, 17, 34, 51, 69, 86, 103, 120]
        );
        assert_eq!(sample_indices(600, 30.).len(), 32);
    }
    #[test]
    fn temporal_pairs_keep_channel_patch_order() {
        let mut a = ImagePatches {
            pixels: vec![1., 1., 2., 2., 3., 3.],
            grid_thw: [1, 1, 1],
            patch_dim: 6,
            merge_size: 1,
        };
        let b = ImagePatches {
            pixels: vec![4., 4., 5., 5., 6., 6.],
            ..a.clone()
        };
        pair_patches(&mut a, &b).unwrap();
        assert_eq!(a.pixels, vec![1., 4., 2., 5., 3., 6.]);
    }
}
