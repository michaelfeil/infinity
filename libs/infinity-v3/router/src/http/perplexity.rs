//! Perplexity embedding contracts on the existing inference queue.
//!
//! Backbones and per-token Dense projections execute in the native backends.
//! Contextual readout pools supplied spans before its FP32 projection.
use super::{model_routing::RoutedJson, response_format::Json, tensor, types::EncodingFormat};
use crate::{ErrorResponse, ModelRuntime, ResponseMetadata};
use anyhow::{ensure, Context};
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use infinity_v3_backend::perplexity::Profile;
use infinity_v3_core::{tokenization::EncodingInput, TextEmbeddingsError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use simsimd::SpatialSimilarity;
use std::{
    collections::HashSet,
    path::{Component, Path},
    sync::Arc,
    time::{Duration, Instant},
};
use tokenizers::{Tokenizer, TruncationDirection};

pub(crate) struct Perplexity {
    profile: Profile,
    tokenizer: Tokenizer,
    literal_tokenizer: Option<Tokenizer>,
    config: Value,
    st: Value,
    lengths: Value,
    skip: HashSet<u32>,
    max_length: usize,
    projection: Option<Projection>,
    mask_token: Option<String>,
    media: Option<infinity_v3_core::multimodal::Qwen3VlProcessor>,
    cpu_workers: Arc<tokio::sync::Semaphore>,
}

struct Projection {
    width: usize,
    weights: Vec<f32>,
}
impl Projection {
    fn load(root: &Path, name: &str, input: usize, output: usize) -> anyhow::Result<Self> {
        let filename = if root.join("model.safetensors").exists() {
            "model.safetensors".to_owned()
        } else {
            let index: Value =
                serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json"))?)?;
            index["weight_map"][name]
                .as_str()
                .context("Missing contextual projection shard")?
                .to_owned()
        };
        ensure!(
            Path::new(&filename).components().count() == 1
                && matches!(
                    Path::new(&filename).components().next(),
                    Some(Component::Normal(_))
                ),
            "Invalid projection shard path"
        );
        ensure!(
            input > 0 && output > 0,
            "Contextual projection dimensions must be positive"
        );
        let file = std::fs::File::open(root.join(filename))?;
        // SAFETY: read-only mapping of a checkpoint held for the duration of decoding.
        let data = unsafe { memmap2::Mmap::map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&data)?;
        let tensor = tensors.tensor(name)?;
        ensure!(
            tensor.shape() == [output, input],
            "Invalid contextual projection geometry"
        );
        // This head is explicitly FP32 in the upstream contract.
        ensure!(
            tensor.dtype() == safetensors::Dtype::F32,
            "Contextual projection must be FP32"
        );
        let weights = tensor
            .data()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        ensure!(
            weights.iter().all(|v| v.is_finite()),
            "Non-finite contextual projection"
        );
        Ok(Self {
            width: input,
            weights,
        })
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, TextEmbeddingsError> {
        if input.len() != self.width {
            return Err(invalid("Contextual hidden width differs from projection"));
        }
        let mut result = super::result_storage::vector(self.weights.len() / self.width)?;
        for row in self.weights.chunks_exact(self.width) {
            result.push(
                f32::dot(row, input).ok_or_else(|| invalid("Contextual projection failed"))? as f32,
            );
        }
        Ok(result)
    }
}

fn read_json(path: &Path) -> anyhow::Result<Value> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
        Err(e) => Err(e.into()),
    }
}

impl Perplexity {
    async fn run_cpu<T: Send + 'static>(
        self: &Arc<Self>,
        work: impl FnOnce(&Self) -> Result<T, TextEmbeddingsError> + Send + 'static,
    ) -> Result<T, TextEmbeddingsError> {
        let worker = self
            .cpu_workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| invalid("Perplexity CPU workers are closed"))?;
        let adapter = self.clone();
        let reservations = infinity_v3_backend::result_memory::REQUEST_RESULTS
            .try_with(Arc::clone)
            .ok();
        tokio::task::spawn_blocking(move || {
            // Keep worker admission and result ownership until the actual work ends,
            // including when the HTTP caller drops its future.
            let _worker = worker;
            match reservations {
                Some(owner) => infinity_v3_backend::result_memory::REQUEST_RESULTS
                    .sync_scope(owner, || work(&adapter)),
                None => work(&adapter),
            }
        })
        .await
        .map_err(|error| {
            TextEmbeddingsError::Backend(infinity_v3_backend::BackendError::Inference(format!(
                "Perplexity CPU worker failed: {error}"
            )))
        })?
    }

    pub(crate) fn capabilities(&self) -> Vec<&'static str> {
        let mut result = vec!["embed", "embed_all", "tokenize", "decode"];
        if self.profile.contextual() {
            result.push("contextualizedembeddings");
        }
        result
    }

    pub(crate) fn multivector(&self) -> bool {
        self.profile.late() || self.profile.contextual()
    }

    async fn prepare_image(
        &self,
        source: &str,
        query: bool,
    ) -> Result<Prepared, TextEmbeddingsError> {
        let media = self
            .media
            .as_ref()
            .ok_or_else(|| invalid("Images require a v2 late-interaction vision export"))?;
        let task = if query { "query" } else { "document" };
        let limit = self.lengths[format!("{task}_length")]
            .as_u64()
            .ok_or_else(|| invalid("Missing image token limit"))? as usize;
        let patches = media
            .prepare_images(vec![source.into()], limit.min(self.max_length))
            .await?;
        let image = patches
            .images
            .into_iter()
            .next()
            .ok_or_else(|| invalid("Image preprocessing returned no image"))?;
        let prefix = self.st["prompts"][task]
            .as_str()
            .ok_or_else(|| invalid("Missing image prefix"))?;
        let mut ids = self.encode(prefix)?.get_ids().to_vec();
        let token = |s: &str| {
            self.tokenizer
                .token_to_id(s)
                .ok_or_else(|| invalid("Missing vision token"))
        };
        ids.push(token("<|vision_start|>")?);
        let start = ids.len();
        ids.resize(start + image.token_count(), token("<|image_pad|>")?);
        ids.push(token("<|vision_end|>")?);
        if ids.len() > limit.min(self.max_length) {
            return Err(invalid("Image exceeds the model input limit"));
        }
        let position_ids =
            infinity_v3_core::multimodal::image_positions(ids.len(), &[(start, &image)])
                .map_err(|e| invalid(&e))?;
        let media = Arc::new(infinity_v3_backend::MultimodalEncoding {
            images: vec![(start, image)],
            audios: vec![],
            position_ids,
            memory: Some(patches.memory),
        });
        Ok(Prepared {
            ids,
            spans: vec![],
            media: Some(media),
        })
    }

    pub(crate) fn load(
        root: &Path,
        mut tokenizer: Tokenizer,
        max_length: usize,
        profile: Profile,
        media_config: infinity_v3_core::multimodal::MultimodalConfig,
    ) -> anyhow::Result<Self> {
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let config = read_json(&root.join("config.json"))?;
        let st = read_json(&root.join("config_sentence_transformers.json"))?;
        let lengths = read_json(&root.join("sentence_bert_config.json"))?;
        let projection = if profile == Profile::ContextV2 {
            Some(Projection::load(
                root,
                "contextual_projection.weight",
                config["text_config"]["hidden_size"]
                    .as_u64()
                    .context("Missing contextual hidden size")? as usize,
                config["embedding_dim"]
                    .as_u64()
                    .context("Missing contextual output size")? as usize,
            )?)
        } else {
            None
        };
        let mask = read_json(&root.join("2_MultiVectorMask/config.json"))?;
        let skip_words = if profile == Profile::LateV2 {
            &mask["skiplist_words"]
        } else {
            &st["skiplist_words"]
        };
        let mut skip = HashSet::new();
        if let Some(words) = skip_words.as_array() {
            for word in words {
                let word = word.as_str().context("Invalid punctuation skip list")?;
                if let Some(id) = tokenizer.token_to_id(word) {
                    skip.insert(id);
                }
            }
        }
        if profile == Profile::LateV2 {
            ensure!(
                mask["keep_only_token_ids"].is_null()
                    && mask["skiplist_tasks"] == serde_json::json!(["document"]),
                "Unsupported multi-vector mask contract"
            );
        }
        let token_config = read_json(&root.join("tokenizer_config.json"))?;
        let mask_token = token_config["mask_token"]
            .as_str()
            .or_else(|| token_config["mask_token"]["content"].as_str())
            .map(str::to_owned);
        let media = if profile == Profile::LateV2 && config["vision_config"].is_object() {
            Some(infinity_v3_core::multimodal::Qwen3VlProcessor::load_images(
                root,
                tokenizer.clone(),
                max_length,
                media_config,
            )?)
        } else {
            None
        };
        let literal_tokenizer = if profile == Profile::ContextV2 {
            let mut literal = tokenizer.clone();
            literal.set_encode_special_tokens(true);
            Some(literal)
        } else {
            None
        };
        Ok(Self {
            cpu_workers: Arc::new(tokio::sync::Semaphore::new(
                num_cpus::get().clamp(1, infinity_v3_core::tokenization::MAX_TOKENIZATION_WORKERS),
            )),
            literal_tokenizer,
            media,
            profile,
            tokenizer,
            config,
            st,
            lengths,
            skip,
            max_length,
            projection,
            mask_token,
        })
    }

    fn encode(&self, text: &str) -> Result<tokenizers::Encoding, TextEmbeddingsError> {
        self.tokenizer
            .encode(text, true)
            .map_err(|e| invalid(&e.to_string()))
    }

    fn prepare(&self, chunks: &[String], query: bool) -> Result<Prepared, TextEmbeddingsError> {
        if chunks.is_empty() {
            return Err(invalid("Each document must contain at least one chunk"));
        }
        if (!self.profile.contextual() || query) && chunks.len() != 1 {
            return Err(invalid("This input requires exactly one string"));
        }
        let (encoding, spans, limit) = match self.profile {
            Profile::Dense => (self.encode(&chunks[0])?, Vec::new(), self.max_length),
            Profile::ContextV1 => {
                let sep = self
                    .tokenizer
                    .token_to_id("<|endoftext|>")
                    .ok_or_else(|| invalid("Missing contextual separator"))?;
                if chunks.iter().any(|c| c.contains("<|endoftext|>")) {
                    return Err(invalid("Chunks must not contain the contextual separator"));
                }
                let encoding = self.encode(&chunks.join("<|endoftext|>"))?;
                let mut spans = Vec::new();
                let mut start = 0;
                for (i, id) in encoding.get_ids().iter().enumerate() {
                    if *id == sep {
                        spans.push((start, i));
                        start = i + 1;
                    }
                }
                spans.push((start, encoding.len()));
                if spans.len() != chunks.len() {
                    return Err(invalid("Tokenizer changed the contextual chunk structure"));
                }
                (encoding, spans, self.max_length)
            }
            Profile::ContextV2 => {
                let prefix = self.config[if query {
                    "query_prefix"
                } else {
                    "document_prefix"
                }]
                .as_str()
                .ok_or_else(|| invalid("Missing contextual prefix"))?;
                let boundary = self.config["boundary_marker"]
                    .as_str()
                    .ok_or_else(|| invalid("Missing contextual boundary"))?;
                // Upstream split_special_tokens=True treats document prefixes and
                // separators as literal text. Query markers are inserted as IDs.
                let tokenizer = self
                    .literal_tokenizer
                    .as_ref()
                    .ok_or_else(|| invalid("Missing contextual literal tokenizer"))?;
                if query {
                    let mut enc = tokenizer
                        .encode(chunks[0].as_str(), true)
                        .map_err(|e| invalid(&e.to_string()))?;
                    let id = self
                        .tokenizer
                        .token_to_id(prefix)
                        .ok_or_else(|| invalid("Missing query marker token"))?;
                    let prefix_encoding = tokenizers::Encoding::new(
                        vec![id],
                        vec![0],
                        vec![prefix.into()],
                        vec![],
                        vec![(0, 0)],
                        vec![1],
                        vec![1],
                        vec![],
                        Default::default(),
                    );
                    enc = tokenizers::Encoding::merge(vec![prefix_encoding, enc], false);
                    let len = enc.len();
                    (
                        enc,
                        vec![(0, len)],
                        self.config["query_length"]
                            .as_u64()
                            .unwrap_or(self.max_length as u64) as usize,
                    )
                } else {
                    let mut text = prefix.to_owned();
                    let mut char_spans = Vec::new();
                    for (i, chunk) in chunks.iter().enumerate() {
                        if i > 0 {
                            text.push_str(boundary);
                        }
                        let start = text.len();
                        text.push_str(chunk);
                        char_spans.push((start, text.len()));
                    }
                    let encoding = tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| invalid(&e.to_string()))?;
                    let limit = self.config["document_length"]
                        .as_u64()
                        .unwrap_or(self.max_length as u64) as usize;
                    check_token_limit(encoding.len(), limit.min(self.max_length))?;
                    let spans = chunk_token_spans(encoding.get_offsets(), &char_spans);
                    (
                        encoding,
                        spans,
                        self.config["document_length"]
                            .as_u64()
                            .unwrap_or(self.max_length as u64) as usize,
                    )
                }
            }
            Profile::LateV1 | Profile::LateV2 => {
                let task = if query { "query" } else { "document" };
                let prefix = if self.profile == Profile::LateV1 {
                    &self.st[format!("{task}_prefix")]
                } else {
                    &self.st["prompts"][task]
                }
                .as_str()
                .ok_or_else(|| invalid("Missing late-interaction prefix"))?;
                let mut encoding = self.encode(&format!("{prefix}{}", chunks[0]))?;
                let length = if self.profile == Profile::LateV1 {
                    &self.st[format!("{task}_length")]
                } else {
                    &self.lengths[format!("{task}_length")]
                }
                .as_u64()
                .ok_or_else(|| invalid("Missing late-interaction length"))?
                    as usize;
                // Query expansion is part of the v1 model input, including its
                // attention mask; it must participate in backbone inference.
                if self.profile == Profile::LateV1 && query && encoding.len() < length {
                    let token = self
                        .mask_token
                        .as_deref()
                        .ok_or_else(|| invalid("Missing query expansion token"))?;
                    let id = self
                        .tokenizer
                        .token_to_id(token)
                        .ok_or_else(|| invalid("Missing query expansion token"))?;
                    let n = length - encoding.len();
                    let expansion = tokenizers::Encoding::new(
                        vec![id; n],
                        vec![0; n],
                        vec![token.into(); n],
                        vec![],
                        vec![(0, 0); n],
                        vec![1; n],
                        vec![1; n],
                        vec![],
                        Default::default(),
                    );
                    encoding = tokenizers::Encoding::merge(vec![encoding, expansion], false);
                }
                (encoding, Vec::new(), length)
            }
        };
        if encoding.is_empty() {
            return Err(invalid("Input produced no tokens"));
        }
        check_token_limit(encoding.len(), limit.min(self.max_length))?;
        let ids = encoding.get_ids().to_vec();
        Ok(Prepared {
            ids,
            spans,
            media: None,
        })
    }

    fn readout(
        &self,
        prepared: &Prepared,
        hidden: Vec<Vec<f32>>,
        request: &ReadoutOptions,
    ) -> Result<Vec<Vec<f32>>, TextEmbeddingsError> {
        if hidden.len() != prepared.ids.len() {
            return Err(invalid("Token readout length mismatch"));
        }
        let mut vectors = super::result_storage::vector(if self.profile.late() {
            hidden.len()
        } else {
            prepared.spans.len()
        })?;
        if self.profile.late() {
            for (id, mut vector) in prepared.ids.iter().zip(hidden) {
                if request.task == PerplexityTask::Document && self.skip.contains(id) {
                    continue;
                }
                normalize(&mut vector);
                vectors.push(vector);
            }
        } else {
            let width = hidden.first().map_or(0, Vec::len);
            // V2 upstream uses a prefix sum before extracting document spans.
            let mut prefix = super::result_storage::vector(width)?;
            prefix.resize(width, 0.);
            let mut starts = std::collections::HashMap::new();
            let points = prepared
                .spans
                .iter()
                .flat_map(|(a, b)| [*a, *b])
                .collect::<HashSet<_>>();
            for i in 0..=hidden.len() {
                if points.contains(&i) {
                    let mut snapshot = super::result_storage::vector(width)?;
                    snapshot.extend_from_slice(&prefix);
                    starts.insert(i, snapshot);
                }
                if i < hidden.len() {
                    for (a, b) in prefix.iter_mut().zip(&hidden[i]) {
                        *a += b;
                    }
                }
            }
            for (start, end) in &prepared.spans {
                let mut vector = super::result_storage::vector(width)?;
                if self.profile == Profile::ContextV2 {
                    vector.extend(starts[end].iter().zip(&starts[start]).map(|(a, b)| a - b));
                    // Divide by the span length after subtracting prefix sums.
                    for v in &mut vector {
                        *v /= (end - start).max(1) as f32;
                    }
                } else {
                    vector.resize(width, 0.);
                    for token in &hidden[*start..*end] {
                        for (a, b) in vector.iter_mut().zip(token) {
                            *a += b;
                        }
                    }
                    for v in &mut vector {
                        *v /= (end - start).max(1) as f32;
                    }
                }
                if let Some(projection) = &self.projection {
                    vector = projection.forward(&vector)?;
                }
                finish(&mut vector, request)?;
                if request.normalize {
                    normalize(&mut vector);
                }
                vectors.push(vector);
            }
        }
        Ok(vectors)
    }
}

struct Prepared {
    ids: Vec<u32>,
    spans: Vec<(usize, usize)>,
    media: Option<Arc<infinity_v3_backend::MultimodalEncoding>>,
}
impl Prepared {
    fn encoding(&self) -> infinity_v3_core::tokenization::ValidEncoding {
        let n = self.ids.len();
        infinity_v3_core::tokenization::ValidEncoding {
            multimodal: self.media.clone(),
            input_ids: self.ids.clone(),
            token_type_ids: vec![0; n],
            position_ids: (0..n as u32).collect(),
            tokens: vec![],
            offsets: vec![],
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PerplexityTask {
    Query,
    #[default]
    Document,
}
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PerplexityQuantization {
    #[default]
    Int8,
    Binary,
    Ubinary,
}
#[derive(Deserialize, utoipa::ToSchema)]
#[serde(untagged)]
pub(crate) enum PerplexityInput {
    Text(String),
    Texts(Vec<String>),
    Documents(Vec<Vec<String>>),
    Image(PerplexityImageInput),
    Images(Vec<PerplexityImageInput>),
}
#[derive(Deserialize, utoipa::ToSchema)]
pub(crate) struct PerplexityImageInput {
    image_url: String,
}

#[derive(Deserialize, utoipa::ToSchema)]
pub(crate) struct PerplexityRequest {
    #[serde(alias = "inputs")]
    input: PerplexityInput,
    #[allow(dead_code)]
    model: Option<String>,
    #[serde(default, alias = "input_type", alias = "prompt_name")]
    task: PerplexityTask,
    #[serde(default)]
    quantization: PerplexityQuantization,
    #[serde(default, alias = "normalize_embeddings")]
    normalize: bool,
    dimensions: Option<usize>,
    #[serde(default)]
    encoding_format: EncodingFormat,
    #[serde(default)]
    truncate: bool,
}

#[derive(Clone, Copy)]
struct ReadoutOptions {
    task: PerplexityTask,
    quantization: PerplexityQuantization,
    normalize: bool,
    dimensions: Option<usize>,
}
impl From<&PerplexityRequest> for ReadoutOptions {
    fn from(request: &PerplexityRequest) -> Self {
        Self {
            task: request.task,
            quantization: request.quantization,
            normalize: request.normalize,
            dimensions: request.dimensions,
        }
    }
}

fn check_token_limit(tokens: usize, limit: usize) -> Result<(), TextEmbeddingsError> {
    if tokens > limit {
        return Err(invalid(
            "Input exceeds the model token limit; inputs are never silently truncated",
        ));
    }
    Ok(())
}

fn chunk_token_spans(offsets: &[(usize, usize)], chunks: &[(usize, usize)]) -> Vec<(usize, usize)> {
    // Tokenizer offsets and document chunks are ordered. Retain original token
    // indices while skipping zero-width special tokens. A token overlapping a
    // boundary may contribute to both chunks, matching upstream late chunking.
    let tokens = offsets
        .iter()
        .enumerate()
        .filter(|(_, (a, b))| b > a)
        .collect::<Vec<_>>();
    let (mut first, mut after) = (0, 0);
    let mut spans = Vec::with_capacity(chunks.len());
    for &(start, end) in chunks {
        if start == end {
            spans.push((0, 0));
            continue;
        }
        while first < tokens.len() && tokens[first].1 .1 <= start {
            first += 1;
        }
        after = after.max(first);
        while after < tokens.len() && tokens[after].1 .0 < end {
            after += 1;
        }
        if first == after {
            spans.push((0, 0));
        } else {
            spans.push((tokens[first].0, tokens[after - 1].0 + 1));
        }
    }
    spans
}

fn invalid(message: &str) -> TextEmbeddingsError {
    TextEmbeddingsError::Validation(message.into())
}
fn validate_quantization(
    profile: Profile,
    quantization: PerplexityQuantization,
) -> Result<(), TextEmbeddingsError> {
    if profile.late() && quantization != PerplexityQuantization::Int8 {
        return Err(invalid(
            "Late-interaction models return normalized floating-point token vectors; binary quantization is unsupported",
        ));
    }
    Ok(())
}

fn resize(vector: &mut Vec<f32>, dimensions: Option<usize>) -> Result<(), TextEmbeddingsError> {
    if let Some(n) = dimensions {
        if n == 0 || n > vector.len() {
            return Err(invalid(
                "dimensions must be positive and no larger than the embedding width",
            ));
        }
        vector.truncate(n);
    }
    Ok(())
}
fn finish(vector: &mut Vec<f32>, request: &ReadoutOptions) -> Result<(), TextEmbeddingsError> {
    quantize(vector, request.quantization);
    resize(vector, request.dimensions)?;
    if request.quantization == PerplexityQuantization::Ubinary {
        let mut packed = super::result_storage::vector(vector.len().div_ceil(8))?;
        for row in vector.chunks(8) {
            let byte = row
                .iter()
                .enumerate()
                .fold(0u8, |byte, (i, v)| byte | (u8::from(*v >= 0.) << (7 - i)));
            packed.push(byte as f32);
        }
        *vector = packed;
    }
    Ok(())
}

fn normalize(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt()
        .max(1e-12) as f32;
    for v in vector {
        *v /= norm;
    }
}
fn quantize(vector: &mut [f32], method: PerplexityQuantization) {
    for v in vector {
        *v = match method {
            PerplexityQuantization::Int8 => (v.tanh() * 127.).round_ties_even().clamp(-128., 127.),
            PerplexityQuantization::Binary | PerplexityQuantization::Ubinary => {
                if *v >= 0. {
                    1.
                } else {
                    -1.
                }
            }
        };
    }
}

#[derive(Serialize)]
struct Envelope {
    object: &'static str,
    data: Vec<Item>,
    model: String,
    usage: Usage,
}
#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}
#[derive(Serialize)]
struct Item {
    object: &'static str,
    index: usize,
    #[serde(flatten)]
    payload: Payload,
}
#[derive(Serialize)]
#[serde(untagged)]
enum Payload {
    Vector {
        embedding: super::types::Embedding,
    },
    Matrix {
        embedding: super::types::AllEmbedding,
    },
    Document {
        data: Vec<Item>,
    },
}

#[utoipa::path(post, tag="infinity_v3", path="/v1/contextualizedembeddings",
    request_body=PerplexityRequest,
    responses((status=200, description="One embedding per supplied chunk, grouped by document"),
              (status=413, description="Invalid input or unsupported model contract",body=crate::ErrorResponse)))]
pub(crate) async fn contextual(RoutedJson { runtime, value }: RoutedJson<Value>) -> Response {
    respond(runtime, value, true, false).await
}

pub(crate) async fn respond(
    runtime: Arc<ModelRuntime>,
    value: Value,
    contextual: bool,
    native: bool,
) -> Response {
    match execute(&runtime, value, contextual, native).await {
        Ok(response) => response,
        Err(error) => {
            let error = ErrorResponse::from(error);
            if native {
                let response: (StatusCode, Json<ErrorResponse>) = error.into();
                response.into_response()
            } else {
                let response: (StatusCode, Json<super::types::OpenAICompatErrorResponse>) =
                    error.into();
                response.into_response()
            }
        }
    }
}

async fn execute(
    runtime: &ModelRuntime,
    value: Value,
    contextual: bool,
    native: bool,
) -> Result<Response, TextEmbeddingsError> {
    let adapter = runtime
        .perplexity
        .as_ref()
        .ok_or_else(|| invalid("This model does not implement Perplexity contextual embeddings"))?;
    let request: PerplexityRequest =
        serde_json::from_value(value).map_err(|e| invalid(&e.to_string()))?;
    validate_quantization(adapter.profile, request.quantization)?;
    if request.truncate {
        return Err(invalid("Perplexity adapters require truncate=false"));
    }
    if request.quantization == PerplexityQuantization::Ubinary && request.normalize {
        return Err(invalid("Packed binary embeddings cannot be normalized"));
    }
    if adapter.profile == Profile::ContextV2 && request.quantization != PerplexityQuantization::Int8
    {
        return Err(invalid("Contextual v2 supports only int8 quantization"));
    }
    if adapter.profile.late() && request.dimensions.is_some() {
        return Err(invalid(
            "Late-interaction models have a fixed 128-dimensional output",
        ));
    }
    enum Document {
        Text(Vec<String>),
        Image(String),
    }
    let documents = match &request.input {
        PerplexityInput::Text(s) => vec![Document::Text(vec![s.clone()])],
        PerplexityInput::Texts(texts) => texts
            .iter()
            .map(|s| Document::Text(vec![s.clone()]))
            .collect(),
        PerplexityInput::Documents(docs) if adapter.profile.contextual() => {
            docs.iter().cloned().map(Document::Text).collect()
        }
        PerplexityInput::Image(image) => vec![Document::Image(image.image_url.clone())],
        PerplexityInput::Images(images) => images
            .iter()
            .map(|image| Document::Image(image.image_url.clone()))
            .collect(),
        _ => return Err(invalid("Nested chunk inputs require a contextual model")),
    };
    if contextual && !adapter.profile.contextual() {
        return Err(invalid("This model is not contextual"));
    }
    if documents.is_empty() || documents.len() > runtime.info.max_client_batch_size {
        return Err(invalid("Invalid document batch size"));
    }
    let mut data = super::result_storage::vector(documents.len())?;
    let mut native_vectors = super::result_storage::vector(documents.len())?;
    let mut native_matrices = super::result_storage::vector(documents.len())?;
    let start_time = Instant::now();
    let options = ReadoutOptions::from(&request);
    let operations = documents.iter().map(|document| async move {
        let permit = runtime.infer.acquire_permit().await;
        let preparation_start = Instant::now();
        let prepared = match document {
            Document::Text(chunks) => {
                let chunks = chunks.clone();
                adapter
                    .run_cpu(move |adapter| {
                        adapter.prepare(&chunks, options.task == PerplexityTask::Query)
                    })
                    .await?
            }
            Document::Image(source) => {
                adapter
                    .prepare_image(source, options.task == PerplexityTask::Query)
                    .await?
            }
        };
        let preparation_time = preparation_start.elapsed();
        let (vectors, metadata) = if adapter.profile == Profile::Dense {
            let response = runtime
                .infer
                .embed_pooled(
                    EncodingInput::Ids(prepared.ids.clone()),
                    false,
                    TruncationDirection::Right,
                    None,
                    false,
                    None,
                    permit,
                    None,
                )
                .await?;
            // A single pooled vector is small; avoid a blocking-pool round trip.
            let mut vector = response.results;
            finish(&mut vector, &options)?;
            if options.normalize {
                normalize(&mut vector);
            }
            (vec![vector], response.metadata)
        } else {
            let response = runtime
                .infer
                .embed_prepared_all(prepared.encoding(), permit)
                .await?;
            (
                adapter
                    .run_cpu(move |adapter| adapter.readout(&prepared, response.results, &options))
                    .await?,
                response.metadata,
            )
        };
        Ok::<_, TextEmbeddingsError>((vectors, metadata, preparation_time))
    });
    let results = futures::future::join_all(operations)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let mut tokens = 0;
    let mut tokenization_time = Duration::ZERO;
    let mut queue_time = Duration::ZERO;
    let mut inference_time = Duration::ZERO;
    for (index, (vectors, metadata, preparation_time)) in results.into_iter().enumerate() {
        tokens += metadata.prompt_tokens;
        tokenization_time += metadata.tokenization + preparation_time;
        queue_time += metadata.queue;
        inference_time += metadata.inference;
        if native {
            if adapter.profile.late() || adapter.profile.contextual() {
                native_matrices.push(tensor::encode_matrix(vectors, request.encoding_format)?);
            } else {
                native_vectors.push(tensor::encode_vector(
                    vectors.into_iter().next().unwrap(),
                    request.encoding_format,
                )?);
            }
        } else if adapter.profile.contextual() {
            let mut items = super::result_storage::vector(vectors.len())?;
            for (index, vector) in vectors.into_iter().enumerate() {
                items.push(Item {
                    object: "embedding",
                    index,
                    payload: Payload::Vector {
                        embedding: tensor::encode_vector(vector, request.encoding_format)?,
                    },
                });
            }
            data.push(Item {
                object: "list",
                index,
                payload: Payload::Document { data: items },
            });
        } else if adapter.profile.late() {
            data.push(Item {
                object: "embedding",
                index,
                payload: Payload::Matrix {
                    embedding: tensor::encode_matrix(vectors, request.encoding_format)?,
                },
            });
        } else {
            data.push(Item {
                object: "embedding",
                index,
                payload: Payload::Vector {
                    embedding: tensor::encode_vector(
                        vectors.into_iter().next().unwrap(),
                        request.encoding_format,
                    )?,
                },
            });
        }
    }
    let chars = documents
        .iter()
        .map(|doc| match doc {
            Document::Text(chunks) => chunks.iter().map(|s| s.chars().count()).sum::<usize>(),
            Document::Image(_) => 0,
        })
        .sum();
    let count = documents.len() as u32;
    let metadata = ResponseMetadata::new(
        chars,
        tokens,
        start_time,
        tokenization_time / count,
        queue_time / count,
        inference_time / count,
    );
    metadata.record_span(&tracing::Span::current());
    metadata.record_metrics(&runtime.info.model_id);
    let headers = axum::http::HeaderMap::from(metadata);
    if native {
        if adapter.profile == Profile::Dense {
            return Ok((headers, Json(native_vectors)).into_response());
        }
        return Ok((headers, Json(native_matrices)).into_response());
    }
    Ok((
        headers,
        Json(Envelope {
            object: "list",
            data,
            model: runtime.info.model_id.clone(),
            usage: Usage {
                prompt_tokens: tokens,
                total_tokens: tokens,
            },
        }),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn adapter(profile: Profile) -> Perplexity {
        Perplexity {
            profile,
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            literal_tokenizer: None,
            config: Value::Null,
            st: Value::Null,
            lengths: Value::Null,
            skip: HashSet::new(),
            max_length: 64,
            projection: None,
            mask_token: None,
            media: None,
            cpu_workers: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }

    fn options(value: Value) -> ReadoutOptions {
        let request: PerplexityRequest = serde_json::from_value(value).unwrap();
        ReadoutOptions::from(&request)
    }

    #[test]
    fn chunk_spans_preserve_overlaps_empty_chunks_and_special_offsets() {
        let offsets = [(0, 0), (0, 2), (2, 5), (2, 5), (5, 9), (0, 0)];
        let chunks = [(0, 3), (3, 3), (3, 7), (9, 9)];
        assert_eq!(
            chunk_token_spans(&offsets, &chunks),
            [(1, 4), (0, 0), (2, 5), (0, 0)]
        );
        assert_eq!(chunk_token_spans(&offsets, &[(9, 10)]), [(0, 0)]);
        assert!(check_token_limit(10, 9).is_err());
        assert!(check_token_limit(9, 9).is_ok());
    }

    #[tokio::test]
    async fn cpu_work_runs_off_http_worker_with_result_accounting() {
        use infinity_v3_backend::result_memory::{RequestResults, ResultPool, REQUEST_RESULTS};
        let adapter = Arc::new(adapter(Profile::ContextV2));
        let pool = ResultPool::new(64).unwrap();
        let owner = Arc::new(RequestResults::default());
        owner.retain(Some(pool.reserve(16).unwrap()));
        let http_thread = std::thread::current().id();
        REQUEST_RESULTS
            .scope(owner.clone(), async {
                adapter
                    .run_cpu(move |_| {
                        assert_ne!(std::thread::current().id(), http_thread);
                        infinity_v3_backend::result_memory::reserve_for_request(16)?;
                        Ok(())
                    })
                    .await
                    .unwrap();
            })
            .await;
        assert_eq!(pool.used(), 32);
        assert_eq!(adapter.cpu_workers.available_permits(), 1);
        drop(owner);
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn late_profiles_reject_binary_quantization_before_inference() {
        for profile in [Profile::LateV1, Profile::LateV2] {
            for value in [
                json!({"input":"query"}),
                json!({"input":"query", "quantization":"int8"}),
            ] {
                assert!(validate_quantization(profile, options(value).quantization).is_ok());
            }
            for quantization in ["binary", "ubinary"] {
                let value = options(json!({"input":"query", "quantization":quantization}));
                assert!(validate_quantization(profile, value.quantization).is_err());
                assert!(validate_quantization(Profile::Dense, value.quantization).is_ok());
                assert!(validate_quantization(Profile::ContextV1, value.quantization).is_ok());
            }
        }
    }

    #[test]
    fn contextual_pooling_excludes_boundaries_and_preserves_empty_chunks() {
        let adapter = adapter(Profile::ContextV1);
        let prepared = Prepared {
            ids: vec![1, 2, 99, 3],
            spans: vec![(0, 2), (3, 3), (3, 4)],
            media: None,
        };
        let actual = adapter
            .readout(
                &prepared,
                vec![vec![-3., 1.], vec![3., 1.], vec![99., 99.], vec![-1., 0.]],
                &options(json!({"input":[["a","","b"]]})),
            )
            .unwrap();
        assert_eq!(actual, vec![vec![0., 97.], vec![0., 0.], vec![-97., 0.]]);
    }

    #[test]
    fn contextual_projection_runs_after_pooling_in_fp32() {
        let mut adapter = adapter(Profile::ContextV2);
        adapter.projection = Some(Projection {
            width: 2,
            weights: vec![2., -1., -1., 2.],
        });
        let prepared = Prepared {
            ids: vec![99, 1, 2],
            spans: vec![(1, 3)],
            media: None,
        };
        let actual = adapter
            .readout(
                &prepared,
                vec![vec![90., 90.], vec![1., 0.], vec![0., 1.]],
                &options(json!({"input":[["a"]]})),
            )
            .unwrap();
        assert_eq!(actual, vec![vec![59., 59.]]);
    }

    #[test]
    fn late_readout_masks_document_punctuation_and_normalizes_each_token() {
        let mut adapter = adapter(Profile::LateV2);
        adapter.skip.insert(99);
        let prepared = Prepared {
            ids: vec![1, 99, 2],
            spans: vec![],
            media: None,
        };
        let hidden = vec![vec![3., 4.], vec![0., 1.], vec![0., 0.]];
        let document = adapter
            .readout(&prepared, hidden.clone(), &options(json!({"input":"a!"})))
            .unwrap();
        assert_eq!(document, vec![vec![0.6, 0.8], vec![0., 0.]]);
        let query = adapter
            .readout(
                &prepared,
                hidden,
                &options(json!({"input":"a!","task":"query"})),
            )
            .unwrap();
        assert_eq!(query.len(), 3);
    }

    #[test]
    fn binary_packing_is_big_endian_and_zero_is_positive() {
        let mut vector = vec![1., -1., 0., -1., 1., -1., 1., -1., 1.];
        finish(
            &mut vector,
            &options(json!({"input":"a","quantization":"ubinary"})),
        )
        .unwrap();
        assert_eq!(vector, vec![170., 128.]);
        let mut zero = vec![0., 0.];
        normalize(&mut zero);
        assert_eq!(zero, vec![0., 0.]);
    }
}
