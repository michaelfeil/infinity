use super::tensor::Tensor;
use crate::ErrorType;
use infinity_v3_core::input::{Message, MessageInput};
use infinity_v3_core::tokenization::EncodingInput;
use serde::de::{value::MapAccessDeserializer, MapAccess, SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::json;
use std::fmt::Formatter;
use utoipa::openapi::{RefOr, Schema};
use utoipa::ToSchema;

fn default_ignore_labels() -> Vec<String> {
    vec!["O".to_string()]
}

use crate::http::ner::AggregationStrategy;

#[derive(Debug)]
pub(crate) enum Sequence {
    Single(String),
    Pair(String, String),
    Ids(Vec<u32>),
    Messages(MessageInput),
}

impl Sequence {
    pub(crate) fn count_chars(&self) -> usize {
        match self {
            Sequence::Single(s) => s.chars().count(),
            Sequence::Pair(s1, s2) => s1.chars().count() + s2.chars().count(),
            Sequence::Ids(_) => 0,
            Sequence::Messages(input) => input
                .messages
                .iter()
                .map(|message| {
                    use infinity_v3_core::input::{ContentPart, MessageContent};
                    match &message.content {
                        MessageContent::Text(text) => text.chars().count(),
                        MessageContent::Parts(parts) => parts
                            .iter()
                            .map(|part| match part {
                                ContentPart::Text { text } => text.chars().count(),
                                _ => 0,
                            })
                            .sum(),
                    }
                })
                .sum(),
        }
    }
}

impl From<Sequence> for EncodingInput {
    fn from(value: Sequence) -> Self {
        match value {
            Sequence::Single(s) => Self::Single(s),
            Sequence::Pair(s1, s2) => Self::Dual(s1, s2),
            Sequence::Ids(ids) => Self::Ids(ids),
            Sequence::Messages(input) => Self::Messages(input.messages),
        }
    }
}

#[derive(Debug)]
pub(crate) enum PredictInput {
    Single(Sequence),
    Batch(Vec<Sequence>),
}

impl<'de> Deserialize<'de> for PredictInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Internal {
            Single(String),
            Multiple(Vec<String>),
            Id(u32),
            Ids(Vec<u32>),
            Messages(MessageInput),
        }

        struct PredictInputVisitor;

        impl<'de> Visitor<'de> for PredictInputVisitor {
            type Value = PredictInput;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter.write_str(
                    "a string, \
                    a pair of strings [string, string] \
                    a batch of mixed strings and pairs [[string], [string, string], ...], \
                    a final token-ID sequence [integer, ...] or a batch [[integer, ...], ...], \
                    a conversation {messages: [...]} or a batch of conversations",
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(PredictInput::Single(Sequence::Single(v.to_string())))
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let input = MessageInput::deserialize(MapAccessDeserializer::new(map))?;
                Ok(PredictInput::Single(Sequence::Messages(input)))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let sequence_from_vec = |mut value: Vec<String>| {
                    // Validate that value is correct
                    match value.len() {
                        1 => Ok(Sequence::Single(value.pop().unwrap())),
                        2 => {
                            // Second element is last
                            let second = value.pop().unwrap();
                            let first = value.pop().unwrap();
                            Ok(Sequence::Pair(first, second))
                        }
                        // Sequence can only be a single string or a pair of strings
                        _ => Err(de::Error::invalid_length(value.len(), &self)),
                    }
                };

                // Get first element
                // This will determine if input is a batch or not
                let s = match seq
                    .next_element::<Internal>()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?
                {
                    // Input is not a batch
                    // Return early
                    Internal::Single(value) => {
                        // Option get second element
                        let second = seq.next_element()?;

                        if seq.next_element::<String>()?.is_some() {
                            // Error as we do not accept > 2 elements
                            return Err(de::Error::invalid_length(3, &self));
                        }

                        if let Some(second) = second {
                            // Second element exists
                            // This is a pair
                            return Ok(PredictInput::Single(Sequence::Pair(value, second)));
                        } else {
                            // Second element does not exist
                            return Ok(PredictInput::Single(Sequence::Single(value)));
                        }
                    }
                    // Input is a batch
                    Internal::Multiple(value) => sequence_from_vec(value),
                    Internal::Id(id) => {
                        let mut ids = vec![id];
                        while let Some(id) = seq.next_element::<u32>()? {
                            ids.push(id);
                        }
                        return Ok(PredictInput::Single(Sequence::Ids(ids)));
                    }
                    Internal::Messages(input) => {
                        let mut batch = vec![Sequence::Messages(input)];
                        while let Some(input) = seq.next_element::<MessageInput>()? {
                            batch.push(Sequence::Messages(input));
                        }
                        return Ok(PredictInput::Batch(batch));
                    }
                    Internal::Ids(ids) => {
                        let mut batch = vec![Sequence::Ids(ids)];
                        while let Some(ids) = seq.next_element::<Vec<u32>>()? {
                            batch.push(Sequence::Ids(ids));
                        }
                        return Ok(PredictInput::Batch(batch));
                    }
                }?;

                let mut batch = Vec::with_capacity(32);
                // Push first sequence
                batch.push(s);

                // Iterate on all sequences
                while let Some(value) = seq.next_element::<Vec<String>>()? {
                    // Validate sequence
                    let s = sequence_from_vec(value)?;
                    // Push to batch
                    batch.push(s);
                }
                Ok(PredictInput::Batch(batch))
            }
        }

        deserializer.deserialize_any(PredictInputVisitor)
    }
}

impl<'__s> ToSchema<'__s> for PredictInput {
    fn schema() -> (&'__s str, RefOr<Schema>) {
        let token_ids = utoipa::openapi::ArrayBuilder::new()
            .items(
                utoipa::openapi::ObjectBuilder::new()
                    .schema_type(utoipa::openapi::SchemaType::Integer)
                    .minimum(Some(0.0))
                    .maximum(Some(u32::MAX as f64)),
            )
            .min_items(Some(1))
            .description(Some(
                "A final, unpadded token-ID sequence; segment IDs are zero",
            ))
            .build();
        (
            "PredictInput",
            utoipa::openapi::OneOfBuilder::new()
                .item(
                    utoipa::openapi::ObjectBuilder::new()
                        .schema_type(utoipa::openapi::SchemaType::String)
                        .description(Some("A single string")),
                )
                .item(
                    utoipa::openapi::ArrayBuilder::new()
                        .items(
                            utoipa::openapi::ObjectBuilder::new()
                                .schema_type(utoipa::openapi::SchemaType::String),
                        )
                        .description(Some("A pair of strings"))
                        .min_items(Some(2))
                        .max_items(Some(2)),
                )
                .item(
                    utoipa::openapi::ArrayBuilder::new()
                        .items(
                            utoipa::openapi::OneOfBuilder::new()
                                .item(
                                    utoipa::openapi::ArrayBuilder::new()
                                        .items(
                                            utoipa::openapi::ObjectBuilder::new()
                                                .schema_type(utoipa::openapi::SchemaType::String),
                                        )
                                        .description(Some("A single string"))
                                        .min_items(Some(1))
                                        .max_items(Some(1)),
                                )
                                .item(
                                    utoipa::openapi::ArrayBuilder::new()
                                        .items(
                                            utoipa::openapi::ObjectBuilder::new()
                                                .schema_type(utoipa::openapi::SchemaType::String),
                                        )
                                        .description(Some("A pair of strings"))
                                        .min_items(Some(2))
                                        .max_items(Some(2)),
                                ),
                        )
                        .description(Some("A batch")),
                )
                .item(utoipa::openapi::Ref::from_schema_name("MessageInput"))
                .item(
                    utoipa::openapi::ArrayBuilder::new()
                        .items(utoipa::openapi::Ref::from_schema_name("MessageInput"))
                        .min_items(Some(1))
                        .description(Some("A batch of independent conversations")),
                )
                .item(token_ids.clone())
                .item(
                    utoipa::openapi::ArrayBuilder::new()
                        .items(token_ids)
                        .min_items(Some(1))
                        .description(Some("A batch of independent final token-ID sequences")),
                )
                .description(Some(
                    "Model input: a string, a string pair, a batch of single strings and pairs, \
                    conversations, or final token IDs and batches of final token IDs.",
                ))
                .example(Some(json!("What is Deep Learning?")))
                .into(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, ToSchema, Eq, Default)]
pub(crate) enum TruncationDirection {
    #[serde(alias = "left", alias = "Left")]
    Left,
    #[serde(alias = "right", alias = "Right")]
    #[default]
    Right,
}

impl From<TruncationDirection> for tokenizers::TruncationDirection {
    fn from(value: TruncationDirection) -> Self {
        match value {
            TruncationDirection::Left => Self::Left,
            TruncationDirection::Right => Self::Right,
        }
    }
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct PredictRequest {
    /// Served model name; required for multi-model deployments.
    pub model: Option<String>,
    pub inputs: PredictInput,
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub raw_scores: bool,
}

/// SGLang-compatible classification, with native messages and truncation extensions.
/// Provide exactly one of input or messages. String arrays are independent inputs.
#[derive(Deserialize, ToSchema)]
pub(crate) struct ClassifyRequest {
    #[serde(default, deserialize_with = "present_input")]
    pub input: Option<EmbeddingInput>,
    #[serde(default, deserialize_with = "present_input")]
    pub messages: Option<Vec<Message>>,
    /// Served model name; required for multi-model deployments.
    #[allow(dead_code)]
    pub model: Option<String>,
    #[allow(dead_code)]
    pub user: Option<String>,
    pub rid: Option<ClassifyRequestId>,
    /// Only absent or zero priority is supported.
    pub priority: Option<i32>,
    pub truncate: Option<bool>,
    #[serde(default)]
    pub truncation_direction: TruncationDirection,
    /// vLLM extension: `false` returns raw logits instead of softmax probabilities, which
    /// one-label cross-encoders need for a usable relevance score.
    #[serde(default)]
    pub use_activation: Option<bool>,
}

#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum ClassifyRequestId {
    Single(String),
    Batch(Vec<String>),
}

impl ClassifyRequest {
    pub fn into_predict(self) -> Result<PredictRequest, &'static str> {
        if self.priority.is_some_and(|priority| priority != 0) {
            return Err("Nonzero classification priority is not supported");
        }
        let input = match (self.input, self.messages) {
            (Some(input), None) => input,
            (None, Some(messages)) => EmbeddingInput::Messages(messages),
            _ => return Err("Provide exactly one of `input` or `messages`"),
        };
        let sequence = |input| match input {
            EncodingInput::Single(text) => Sequence::Single(text),
            EncodingInput::Ids(ids) => Sequence::Ids(ids),
            EncodingInput::Messages(messages) => Sequence::Messages(MessageInput { messages }),
            EncodingInput::Dual(_, _) => unreachable!("embedding inputs cannot contain pairs"),
        };
        let inputs = match InputBatch::from(input) {
            InputBatch::Single(input) => PredictInput::Single(sequence(input)),
            InputBatch::Batch(inputs) => {
                if inputs.is_empty() {
                    return Err("Input cannot be empty");
                }
                PredictInput::Batch(inputs.into_iter().map(sequence).collect())
            }
        };
        let empty = |input: &Sequence| match input {
            Sequence::Single(text) => text.trim().is_empty(),
            Sequence::Ids(ids) => ids.is_empty(),
            Sequence::Messages(input) => input.messages.is_empty(),
            Sequence::Pair(_, _) => false,
        };
        if match &inputs {
            PredictInput::Single(input) => empty(input),
            PredictInput::Batch(inputs) => inputs.iter().any(empty),
        } {
            return Err("Input cannot be empty or whitespace only");
        }
        let count = match &inputs {
            PredictInput::Single(_) => 1,
            PredictInput::Batch(inputs) => inputs.len(),
        };
        // Request IDs are accepted as metadata; response IDs are independently generated.
        match self.rid {
            Some(ClassifyRequestId::Batch(ids)) if ids.len() != count => {
                return Err("rid batch length must match the input batch")
            }
            Some(ClassifyRequestId::Single(id)) => drop(id),
            _ => (),
        }
        Ok(PredictRequest {
            model: None,
            inputs,
            truncate: self.truncate,
            truncation_direction: self.truncation_direction,
            // SGLang applies softmax even to one-label classifiers.
            raw_scores: true,
        })
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ClassifyData {
    pub index: usize,
    pub label: String,
    pub probs: Vec<f32>,
    pub num_classes: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ClassifyUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub prompt_tokens_details: Option<serde_json::Value>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ClassifyResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub data: Vec<ClassifyData>,
    pub usage: ClassifyUsage,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ClassifyErrorResponse {
    pub object: &'static str,
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<&'static str>,
    pub code: u16,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct PredictTokensRequest {
    /// Served model name; required for multi-model deployments.
    pub model: Option<String>,
    pub inputs: Vec<String>,
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub raw_scores: bool,
    #[serde(default)]
    #[schema(default = "none", example = "none")]
    pub aggregation_strategy: AggregationStrategy,
    #[serde(default = "default_ignore_labels")]
    pub ignore_labels: Vec<String>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Prediction {
    #[schema(example = "0.5")]
    pub score: f32,
    #[schema(example = "admiration")]
    pub label: String,
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum PredictResponse {
    Single(Vec<Prediction>),
    Batch(Vec<Vec<Prediction>>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct RerankRequest {
    /// Served model name; required for multi-model deployments.
    pub model: Option<String>,
    #[schema(example = "What is Deep Learning?")]
    pub query: String,
    #[schema(example = json!(["Deep Learning is ..."]))]
    pub texts: Vec<String>,
    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub raw_scores: bool,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub return_text: bool,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Rank {
    #[schema(example = "0")]
    pub index: usize,
    #[schema(nullable = true, example = "Deep Learning is ...", default = "null")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[schema(example = "1.0")]
    pub score: f32,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct RerankResponse(pub Vec<Rank>);

pub(crate) use infinity_v3_core::input::EmbeddingInput;

/// Internal batching shape. A conversation remains a single model input.
pub(crate) enum InputBatch {
    Single(EncodingInput),
    Batch(Vec<EncodingInput>),
}

impl From<EmbeddingInput> for InputBatch {
    fn from(value: EmbeddingInput) -> Self {
        match value {
            EmbeddingInput::Text(text) => Self::Single(EncodingInput::Single(text)),
            EmbeddingInput::TextBatch(texts) => {
                Self::Batch(texts.into_iter().map(EncodingInput::Single).collect())
            }
            EmbeddingInput::Messages(messages) => Self::Single(EncodingInput::Messages(messages)),
            EmbeddingInput::TokenIds(ids) => Self::Single(EncodingInput::Ids(ids)),
            EmbeddingInput::TokenIdsBatch(batch) => {
                Self::Batch(batch.into_iter().map(EncodingInput::Ids).collect())
            }
        }
    }
}

/// Embedding representation, not model compute precision. `tensor` returns portable
/// little-endian float32 with explicit shape and dtype: JSON base64 or MessagePack bin.
/// `float` keeps numeric arrays; legacy `base64` keeps native-endian float32 vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, ToSchema, Default)]
#[serde(rename_all = "snake_case")]
#[schema(example = "tensor")]
pub(crate) enum EncodingFormat {
    #[default]
    Float,
    Base64,
    Tensor,
}

#[derive(Deserialize)]
#[serde(try_from = "OpenAICompatRequestBody")]
pub(crate) struct OpenAICompatRequest {
    pub input: EmbeddingInput,
    #[allow(dead_code)]
    pub model: Option<String>,
    #[allow(dead_code)]
    pub user: Option<String>,
    pub encoding_format: EncodingFormat,
    pub dimensions: Option<usize>,
}

#[derive(Deserialize, ToSchema)]
struct OpenAICompatRequestBody {
    /// Provide exactly one of `input` or `messages`.
    #[serde(default, deserialize_with = "present_input")]
    pub input: Option<EmbeddingInput>,
    #[serde(default, deserialize_with = "present_input")]
    #[schema(min_items = 1)]
    pub messages: Option<Vec<Message>>,
    #[allow(dead_code)]
    #[schema(nullable = true, example = "null")]
    pub model: Option<String>,
    #[allow(dead_code)]
    #[schema(nullable = true, example = "null")]
    pub user: Option<String>,
    /// Opt in with "tensor" for portable float32 buffers. Add Accept: application/msgpack
    /// for binary data; JSON returns base64. The default "float" keeps numeric arrays.
    #[schema(default = "float", example = "tensor")]
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    #[schema(default = "null", example = "null", nullable = true)]
    pub dimensions: Option<usize>,
}

// Explicit null is invalid; absent fields remain None via serde(default).
fn present_input<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

impl TryFrom<OpenAICompatRequestBody> for OpenAICompatRequest {
    type Error = &'static str;

    fn try_from(body: OpenAICompatRequestBody) -> Result<Self, Self::Error> {
        let input = match (body.input, body.messages) {
            (Some(input), None) => input,
            (None, Some(messages)) => EmbeddingInput::Messages(messages),
            _ => return Err("Provide exactly one of `input` or `messages`"),
        };
        Ok(Self {
            input,
            model: body.model,
            user: body.user,
            encoding_format: body.encoding_format,
            dimensions: body.dimensions,
        })
    }
}

impl<'s> ToSchema<'s> for OpenAICompatRequest {
    fn schema() -> (&'s str, RefOr<Schema>) {
        let RefOr::T(Schema::Object(mut completion)) = OpenAICompatRequestBody::schema().1 else {
            unreachable!("request body is an object");
        };
        completion.properties.insert(
            "input".into(),
            utoipa::openapi::Ref::from_schema_name("EmbeddingInput").into(),
        );
        completion.properties.insert(
            "messages".into(),
            utoipa::openapi::ArrayBuilder::new()
                .items(utoipa::openapi::Ref::from_schema_name("Message"))
                .min_items(Some(1))
                .build()
                .into(),
        );
        let mut chat = completion.clone();
        completion.required.push("input".into());
        chat.required.push("messages".into());
        (
            "OpenAICompatRequest",
            utoipa::openapi::OneOfBuilder::new()
                .item(completion)
                .item(chat)
                .description(Some(
                    "Provide exactly one of input or messages; a message list is one conversation.",
                ))
                .into(),
        )
    }
}

// Share the property definitions once; oneOf only expresses exclusive required fields.
pub(super) fn exclusive_input_schema(schema: RefOr<Schema>) -> RefOr<Schema> {
    let RefOr::T(Schema::Object(mut body)) = schema else {
        unreachable!("request body is an object");
    };
    body.properties.insert(
        "input".into(),
        utoipa::openapi::Ref::from_schema_name("EmbeddingInput").into(),
    );
    body.properties.insert(
        "messages".into(),
        utoipa::openapi::ArrayBuilder::new()
            .items(utoipa::openapi::Ref::from_schema_name("Message"))
            .min_items(Some(1))
            .build()
            .into(),
    );
    utoipa::openapi::AllOfBuilder::new()
        .item(body)
        .item(
            utoipa::openapi::OneOfBuilder::new()
                .item(utoipa::openapi::ObjectBuilder::new().required("input"))
                .item(utoipa::openapi::ObjectBuilder::new().required("messages")),
        )
        .description(Some(
            "Provide exactly one of input or messages; a message list is one conversation.",
        ))
        .into()
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum Embedding {
    Float(Vec<f32>),
    All(AllEmbedding),
    Base64(String),
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum AllEmbedding {
    Float(Vec<Vec<f32>>),
    Base64(Vec<String>),
    Tensor(Tensor),
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatEmbedding {
    #[schema(example = "embedding")]
    pub object: &'static str,
    #[schema(example = json!([0.0, 1.0, 2.0]))]
    pub embedding: Embedding,
    #[schema(example = "0")]
    pub index: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatUsage {
    #[schema(example = "512")]
    pub prompt_tokens: usize,
    #[schema(example = "512")]
    pub total_tokens: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatResponse {
    #[schema(example = "list")]
    pub object: &'static str,
    pub data: Vec<OpenAICompatEmbedding>,
    #[schema(example = "thenlper/gte-base")]
    pub model: String,
    pub usage: OpenAICompatUsage,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct SimilarityInput {
    /// The string that you wish to compare the other strings with. This can be a phrase, sentence,
    /// or longer passage, depending on the model being used.
    #[schema(example = "What is Deep Learning?")]
    pub source_sentence: String,
    /// A list of strings which will be compared against the source_sentence.
    #[schema(example = json!(["What is Machine Learning?"]))]
    pub sentences: Vec<String>,
}

#[derive(Deserialize, ToSchema, Default)]
pub(crate) struct SimilarityParameters {
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct SimilarityRequest {
    /// Served model name; required for multi-model deployments.
    pub model: Option<String>,
    pub inputs: SimilarityInput,
    /// Additional inference parameters for Sentence Similarity
    #[schema(default = "null", example = "null", nullable = true)]
    pub parameters: Option<SimilarityParameters>,
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([0.0, 1.0, 0.5]))]
pub(crate) struct SimilarityResponse(pub Vec<f32>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedRequest {
    /// Opt in with "tensor" for portable float32 buffers. Add Accept: application/msgpack
    /// for binary data; JSON returns base64. The default "float" keeps numeric arrays.
    #[serde(default)]
    #[schema(default = "float", example = "tensor")]
    pub encoding_format: EncodingFormat,
    /// Served model name; required for multi-model deployments.
    pub model: Option<String>,
    pub inputs: EmbeddingInput,

    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,

    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,

    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,

    #[serde(default = "default_normalize")]
    #[schema(default = "true", example = "true")]
    pub normalize: bool,

    /// The number of dimensions that the output embeddings should have. If not set, the original
    /// shape of the representation will be returned instead.
    #[schema(default = "null", example = "null", nullable = true)]
    pub dimensions: Option<usize>,
}

fn default_normalize() -> bool {
    true
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([[0.0, 1.0, 2.0]]))]
pub(crate) struct EmbedResponse(pub Vec<Embedding>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedSparseRequest {
    /// Served model name; required for multi-model deployments.
    pub model: Option<String>,
    pub inputs: EmbeddingInput,
    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct SparseValue {
    pub index: usize,
    pub value: f32,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct EmbedSparseResponse(pub Vec<Vec<SparseValue>>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedAllRequest {
    /// Opt in with "tensor" for portable float32 buffers. Add Accept: application/msgpack
    /// for binary data; JSON returns base64. The default "float" keeps numeric arrays.
    #[serde(default)]
    #[schema(default = "float", example = "tensor")]
    pub encoding_format: EncodingFormat,
    /// Served model name; required for multi-model deployments.
    pub model: Option<String>,
    pub inputs: EmbeddingInput,
    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([[[0.0, 1.0, 2.0]]]))]
pub(crate) struct EmbedAllResponse(pub Vec<AllEmbedding>);

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatErrorResponse {
    pub message: String,
    pub code: u16,
    #[serde(rename(serialize = "type"))]
    pub error_type: ErrorType,
}

#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum TokenizeInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct TokenizeRequest {
    /// Served model name; required for multi-model deployments.
    pub model: Option<String>,
    pub inputs: TokenizeInput,
    #[serde(default = "default_add_special_tokens")]
    #[schema(default = "true", example = "true")]
    pub add_special_tokens: bool,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

fn default_add_special_tokens() -> bool {
    true
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct SimpleToken {
    #[schema(example = 0)]
    pub id: u32,
    #[schema(example = "test")]
    pub text: String,
    #[schema(example = "false")]
    pub special: bool,
    #[schema(example = 0)]
    pub start: Option<usize>,
    #[schema(example = 2)]
    pub stop: Option<usize>,
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([[{"id": 0, "text": "test", "special": false, "start": 0, "stop": 2}]]))]
pub(crate) struct TokenizeResponse(pub Vec<Vec<SimpleToken>>);

#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum InputIds {
    Single(Vec<u32>),
    Batch(Vec<Vec<u32>>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct DecodeRequest {
    /// Served model name; required for multi-model deployments.
    pub model: Option<String>,
    pub ids: InputIds,
    #[serde(default = "default_skip_special_tokens")]
    #[schema(default = "true", example = "true")]
    pub skip_special_tokens: bool,
}

fn default_skip_special_tokens() -> bool {
    true
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!(["test"]))]
pub(crate) struct DecodeResponse(pub Vec<String>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct VertexRequest {
    /// Served model name; required for multi-model deployments.
    pub model: Option<String>,
    pub instances: Vec<serde_json::Value>,
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum VertexPrediction {
    Embed(EmbedResponse),
    EmbedSparse(EmbedSparseResponse),
    Predict(PredictResponse),
    Rerank(RerankResponse),
}

#[derive(Serialize, ToSchema)]
pub(crate) struct VertexResponse {
    pub predictions: Vec<VertexPrediction>,
}

#[derive(Serialize, ToSchema, Clone)]
pub(crate) struct TokenPrediction {
    #[schema(example = "Hello")]
    pub token: String,
    #[schema(example = 0)]
    pub token_id: u32,
    #[schema(example = 0)]
    pub start: Option<usize>,
    #[schema(example = 5)]
    pub end: Option<usize>,
    #[schema(example = json!({"O": 9.41, "B-MISC": -1.15, "I-MISC": -0.85}))]
    #[schema(value_type = std::collections::HashMap<String, f32>)]
    #[serde(serialize_with = "serialize_token_scores")]
    pub results: Vec<(String, f32)>,
}

fn serialize_token_scores<S: serde::Serializer>(
    scores: &[(String, f32)],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(Some(scores.len()))?;
    for (label, score) in scores {
        map.serialize_entry(label, score)?;
    }
    map.end()
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum TokenPredictResponse {
    Batch(Vec<Vec<TokenPrediction>>),
}

#[cfg(test)]
mod embedding_input_tests {
    use super::*;

    #[test]
    fn conversation_is_one_input_and_texts_remain_a_batch() {
        let value = json!([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"}
        ]);
        let req: EmbedRequest = serde_json::from_value(json!({"inputs": value})).unwrap();
        let InputBatch::Single(EncodingInput::Messages(messages)) = InputBatch::from(req.inputs)
        else {
            panic!("a conversation must be one input");
        };
        assert_eq!(serde_json::to_value(messages).unwrap(), value);
        let req: OpenAICompatRequest =
            serde_json::from_value(json!({"input": ["hello", "world"]})).unwrap();
        let InputBatch::Batch(inputs) = InputBatch::from(req.input) else {
            panic!("independent texts must stay a batch");
        };
        assert_eq!(inputs.len(), 2);
        assert!(matches!(&inputs[0], EncodingInput::Single(text) if text == "hello"));
        assert!(matches!(&inputs[1], EncodingInput::Single(text) if text == "world"));
    }

    #[test]
    fn token_ids_are_one_input_or_an_ordered_batch_across_endpoints() {
        let req: EmbedRequest = serde_json::from_value(json!({"inputs": [101, 42, 102]})).unwrap();
        assert!(matches!(InputBatch::from(req.inputs),
            InputBatch::Single(EncodingInput::Ids(ids)) if ids == vec![101, 42, 102]));
        let req: OpenAICompatRequest =
            serde_json::from_value(json!({"input": [[101, 42, 102], [101, 43, 102]]})).unwrap();
        let InputBatch::Batch(inputs) = InputBatch::from(req.input) else {
            panic!("token sequences must remain independent inputs");
        };
        assert_eq!(inputs.len(), 2);
        assert!(matches!(&inputs[0], EncodingInput::Ids(ids) if ids == &vec![101, 42, 102]));
        assert!(matches!(&inputs[1], EncodingInput::Ids(ids) if ids == &vec![101, 43, 102]));
        let parse = |value| serde_json::from_value::<PredictInput>(value);
        assert!(matches!(parse(json!([101, 42, 102])).unwrap(),
            PredictInput::Single(Sequence::Ids(ids)) if ids == vec![101, 42, 102]));
        assert!(matches!(parse(json!([[101, 42], [102]])).unwrap(),
            PredictInput::Batch(batch) if batch.len() == 2));
        assert!(matches!(
            parse(json!(["query", "document"])).unwrap(),
            PredictInput::Single(Sequence::Pair(_, _))
        ));
        assert!(parse(json!([[101], ["text"]])).is_err());
    }

    #[test]
    fn openai_messages_are_exclusive_and_preserve_embedding_options() {
        let messages = json!([
            {"role": "system", "content": "Represent this conversation"},
            {"role": "user", "content": "hello"}
        ]);
        let req: OpenAICompatRequest = serde_json::from_value(json!({
            "messages": messages, "encoding_format": "base64", "dimensions": 32,
            "model": "example", "user": "client"
        }))
        .unwrap();
        assert!(matches!(req.encoding_format, EncodingFormat::Base64));
        assert_eq!(req.dimensions, Some(32));
        let InputBatch::Single(EncodingInput::Messages(actual)) = InputBatch::from(req.input)
        else {
            panic!("messages must describe one input");
        };
        assert_eq!(serde_json::to_value(actual).unwrap(), messages);
        for value in [
            json!({}),
            json!({"input": "text", "messages": messages}),
            json!({"input": null, "messages": messages}),
            json!({"input": "text", "messages": null}),
            json!({"messages": "text"}),
            json!({"messages": [101, 42]}),
            json!({"messages": [{"role": "tool", "content": "unsupported"}]}),
        ] {
            assert!(
                serde_json::from_value::<OpenAICompatRequest>(value.clone()).is_err(),
                "{value}"
            );
        }
        // Existing input-based conversations remain supported.
        let req: OpenAICompatRequest = serde_json::from_value(json!({"input": messages})).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Messages(_)));
    }

    #[test]
    fn classification_conversations_are_single_or_independent_batches() {
        let conversation = json!({"messages": [
            {"role": "system", "content": "label"},
            {"role": "user", "content": [{"type": "text", "text": "café"}]}
        ]});
        let req: PredictRequest = serde_json::from_value(json!({"inputs": conversation})).unwrap();
        let PredictInput::Single(input) = req.inputs else {
            panic!("one conversation");
        };
        assert_eq!(input.count_chars(), 9);
        assert!(
            matches!(EncodingInput::from(input), EncodingInput::Messages(messages) if messages.len() == 2)
        );
        let req: PredictRequest = serde_json::from_value(json!({
            "inputs": [conversation, conversation], "raw_scores": true, "truncate": true
        }))
        .unwrap();
        assert!(req.raw_scores);
        assert_eq!(req.truncate, Some(true));
        let PredictInput::Batch(batch) = req.inputs else {
            panic!("independent conversations");
        };
        assert_eq!(batch.len(), 2);
        assert!(batch
            .into_iter()
            .all(|input| matches!(input, Sequence::Messages(_))));
        for inputs in [
            json!({"messages": [], "extra": true}),
            json!([conversation, ["text"]]),
            json!([["text"], conversation]),
            json!({"messages": [{"role": "user", "content": "text", "extra": true}]}),
        ] {
            assert!(serde_json::from_value::<PredictRequest>(json!({"inputs": inputs})).is_err());
        }
    }

    #[test]
    fn openai_schema_describes_both_exclusive_input_shapes() {
        let schema = serde_json::to_value(OpenAICompatRequest::schema().1).unwrap();
        let branches = schema["oneOf"].as_array().unwrap();
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0]["required"], json!(["input"]));
        assert_eq!(
            branches[0]["properties"]["input"]["$ref"],
            "#/components/schemas/EmbeddingInput"
        );
        assert_eq!(branches[0]["properties"], branches[1]["properties"]);
        assert_eq!(branches[1]["required"], json!(["messages"]));
        assert_eq!(
            branches[1]["properties"]["messages"]["items"]["$ref"],
            "#/components/schemas/Message"
        );

        let schema = serde_json::to_value(PredictInput::schema().1).unwrap();
        assert!(schema["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["$ref"] == "#/components/schemas/MessageInput"));
    }

    #[test]
    fn pooling_schemas_match_checked_in_documentation() {
        let documentation: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/openapi.json")).unwrap();
        for (name, schema) in [
            OpenAICompatRequest::schema(),
            PredictInput::schema(),
            MessageInput::schema(),
        ] {
            assert_eq!(
                documentation["components"]["schemas"][name],
                serde_json::to_value(schema).unwrap(),
                "{name}"
            );
        }
    }

    #[test]
    fn embedding_schemas_reference_the_shared_contract() {
        for (schema, field) in [
            (EmbedRequest::schema().1, "inputs"),
            (EmbedAllRequest::schema().1, "inputs"),
            (EmbedSparseRequest::schema().1, "inputs"),
        ] {
            let value = serde_json::to_value(schema).unwrap();
            assert_eq!(
                value["properties"][field]["$ref"],
                "#/components/schemas/EmbeddingInput"
            );
        }
    }
}

#[cfg(test)]
mod classify_request_tests {
    use super::*;

    fn parse(value: serde_json::Value) -> Result<PredictRequest, String> {
        serde_json::from_value::<ClassifyRequest>(value)
            .map_err(|error| error.to_string())?
            .into_predict()
            .map_err(str::to_owned)
    }

    #[test]
    fn sglang_string_arrays_are_batches_not_pairs() {
        let request = parse(json!({"model":"model","input":["first","second"],"user":"client","rid":["a","b"],"priority":0})).unwrap();
        assert!(request.raw_scores);
        let PredictInput::Batch(batch) = request.inputs else {
            panic!("string arrays are batches");
        };
        assert!(matches!(&batch[0], Sequence::Single(text) if text == "first"));
        assert!(matches!(&batch[1], Sequence::Single(text) if text == "second"));
        assert_eq!(batch.len(), 2);
        assert!(matches!(
            parse(json!({"input":"single","rid":"a"})).unwrap().inputs,
            PredictInput::Single(Sequence::Single(_))
        ));
    }

    #[test]
    fn token_ids_and_native_chat_use_existing_inference_inputs() {
        assert!(
            matches!(parse(json!({"input":[101,42,102]})).unwrap().inputs, PredictInput::Single(Sequence::Ids(ids)) if ids == vec![101,42,102])
        );
        assert!(
            matches!(parse(json!({"input":[[101,42],[101,43]]})).unwrap().inputs, PredictInput::Batch(inputs) if inputs.len() == 2)
        );
        let messages =
            json!([{"role":"system","content":"instructions"},{"role":"user","content":"hello"}]);
        for request in [json!({"messages":messages}), json!({"input":messages})] {
            assert!(
                matches!(parse(request).unwrap().inputs, PredictInput::Single(Sequence::Messages(input)) if input.messages.len() == 2)
            );
        }
        let request =
            parse(json!({"input":"hello","truncate":true,"truncation_direction":"left"})).unwrap();
        assert_eq!(request.truncate, Some(true));
        assert_eq!(request.truncation_direction, TruncationDirection::Left);
    }

    #[test]
    fn invalid_inputs_are_rejected_before_inference() {
        let messages = json!([{"role":"user","content":"hello"}]);
        for value in [
            json!({}),
            json!({"input":null}),
            json!({"input":[]}),
            json!({"input":" "}),
            json!({"input":["hello"," "]}),
            json!({"input":[[]]}),
            json!({"messages":[]}),
            json!({"input":"hello","messages":messages}),
            json!({"input":"hello","messages":null}),
            json!({"input":null,"messages":messages}),
            json!({"input":[-1]}),
            json!({"input":[1.5]}),
            json!({"input":"hello","priority":1}),
            json!({"input":["hello","world"],"rid":["a"]}),
        ] {
            assert!(parse(value.clone()).is_err(), "{value}");
        }
    }
}

impl super::model_routing::RequestedModel for PredictRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for PredictTokensRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for RerankRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for SimilarityRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for EmbedRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for EmbedSparseRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for EmbedAllRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for TokenizeRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for DecodeRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for VertexRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for ClassifyRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
impl super::model_routing::RequestedModel for OpenAICompatRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
