//! Shared model input content for embedding and decision adapters.
//!
//! These types describe the request only. Model adapters validate supported roles and
//! modalities and own chat templates and preprocessing. Deserialization never fetches
//! media, renders a conversation, or implies that a loaded model supports its content.
use serde::{Deserialize, Serialize};
#[cfg(feature = "openapi")]
use utoipa::ToSchema;

/// A plain text state or an explicitly structured native conversation.
/// Message objects are never interpreted as text or silently flattened.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(untagged)]
pub enum ModelInput {
    Text(String),
    Messages(MessageInput),
}

/// Embedding input: text, final token IDs, batches of either, or one conversation.
/// A message list always describes one input, regardless of the number of turns.
/// An empty array is parsed as an empty text batch and rejected by the request handler.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(untagged)]
pub enum EmbeddingInput {
    Text(String),
    TextBatch(Vec<String>),
    TokenIds(Vec<u32>),
    TokenIdsBatch(Vec<Vec<u32>>),
    Messages(Vec<Message>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(deny_unknown_fields)]
pub struct MessageInput {
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: MessageRole,
    pub content: MessageContent,
}

/// Roles supported by the initial conversation contract. Additional roles require
/// explicit metadata and adapter support, rather than being silently discarded.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    Developer,
    User,
    Assistant,
    /// Query side of a cross-encoder pair, for checkpoints whose template defines it.
    Query,
    /// Document side of a cross-encoder pair, for checkpoints whose template defines it.
    Document,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// Ordered wire content, independent of a particular model's tensor format.
/// These types describe inputs; the loaded adapter must explicitly support each
/// modality. VideoUrl is an extension following the vLLM message convention.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(from = "ContentPartInput")]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageSource },
    InputAudio { input_audio: AudioSource },
    VideoUrl { video_url: VideoSource },
}

/// Accepted wire spellings normalize before model validation and rendering.
/// Serialization continues to use the existing ContentPart spellings.
#[derive(Deserialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ContentPartInput {
    Text {
        text: String,
    },
    InputText {
        text: String,
    },
    ImageUrl {
        image_url: ImageSource,
    },
    InputImage {
        image_url: String,
        #[serde(default)]
        detail: Option<ImageDetail>,
    },
    InputAudio {
        input_audio: AudioSource,
    },
    VideoUrl {
        video_url: VideoSource,
    },
}

impl From<ContentPartInput> for ContentPart {
    fn from(part: ContentPartInput) -> Self {
        match part {
            ContentPartInput::Text { text } | ContentPartInput::InputText { text } => {
                Self::Text { text }
            }
            ContentPartInput::ImageUrl { image_url } => Self::ImageUrl { image_url },
            ContentPartInput::InputImage { image_url, detail } => Self::ImageUrl {
                image_url: ImageSource {
                    url: image_url,
                    detail,
                },
            },
            ContentPartInput::InputAudio { input_audio } => Self::InputAudio { input_audio },
            ContentPartInput::VideoUrl { video_url } => Self::VideoUrl { video_url },
        }
    }
}

#[cfg(feature = "openapi")]
impl<'s> ToSchema<'s> for ContentPart {
    fn schema() -> (
        &'s str,
        utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
    ) {
        ("ContentPart", ContentPartInput::schema().1)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(deny_unknown_fields)]
pub struct ImageSource {
    /// Remote URL or inline data URL; fetching is a processor responsibility.
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(deny_unknown_fields)]
pub struct AudioSource {
    /// Base64 encoded bytes. Not decoded by request deserialization.
    pub data: String,
    pub format: AudioFormat,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    Wav,
    Mp3,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(ToSchema))]
#[serde(deny_unknown_fields)]
pub struct VideoSource {
    pub url: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_content_parts_normalize_to_legacy_model_input() {
        for url in [
            "https://example.com/photo.png",
            "data:image/png;base64,AA==",
        ] {
            for detail in [None, Some("auto"), Some("low"), Some("high")] {
                let mut image = json!({"type": "input_image", "image_url": url});
                let mut source = json!({"url": url});
                if let Some(detail) = detail {
                    image["detail"] = json!(detail);
                    source["detail"] = json!(detail);
                }
                let aliased = json!({"messages": [{"role": "user", "content": [
                    {"type": "input_text", "text": "Before"}, image,
                    {"type": "text", "text": "Between"},
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "input_text", "text": "After"}
                ]}, {"role": "assistant", "content": "Previous response"}]});
                let expected = json!({"messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Before"},
                    {"type": "image_url", "image_url": source},
                    {"type": "text", "text": "Between"},
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": "After"}
                ]}, {"role": "assistant", "content": "Previous response"}]});
                let input: ModelInput = serde_json::from_value(aliased).unwrap();
                assert_eq!(serde_json::to_value(&input).unwrap(), expected);
                let legacy: ModelInput = serde_json::from_value(expected).unwrap();
                // OneJev renders this canonical representation as part of its state.
                assert_eq!(
                    serde_json::to_string_pretty(&input).unwrap(),
                    serde_json::to_string_pretty(&legacy).unwrap()
                );
            }
        }
    }

    #[test]
    fn content_part_aliases_reject_unsupported_or_ambiguous_fields() {
        for part in [
            json!({"type": "input_image", "file_id": "file-123"}),
            json!({"type": "input_image", "image_url": "x", "file_id": "file-123"}),
            json!({"type": "input_image", "url": "x"}),
            json!({"type": "input_image", "image_url": {"url": "x"}}),
            json!({"type": "input_image", "image_url": "x", "detail": "unsupported"}),
            json!({"type": "image_url", "image_url": "x"}),
            json!({"type": "image_url", "image_url": {"url": "x", "extra": true}}),
            json!({"type": "input_text"}),
            json!({"type": "input_text", "text": "x", "extra": true}),
        ] {
            assert!(
                serde_json::from_value::<ContentPart>(part.clone()).is_err(),
                "{part}"
            );
            let state = json!({"messages": [{"role": "user", "content": [part]}]});
            assert!(serde_json::from_value::<ModelInput>(state).is_err());
        }
    }

    #[cfg(feature = "openapi")]
    #[test]
    fn content_part_schema_matches_published_contract() {
        let checked_in: serde_json::Value =
            serde_json::from_str(include_str!("../../docs/openapi.json")).unwrap();
        assert_eq!(
            serde_json::to_value(ContentPart::schema().1).unwrap(),
            checked_in["components"]["schemas"]["ContentPart"]
        );
    }

    #[test]
    fn embedding_shapes_are_unambiguous() {
        for value in [
            json!("hello"),
            json!(["first", "second"]),
            json!([]),
            json!([1, 2]),
            json!([[1, 2], [3]]),
            json!([
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "https://bucket.example/image?signature=keep"}}
                ]},
                {"role": "assistant", "content": "An image"}
            ]),
        ] {
            let parsed: EmbeddingInput = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(serde_json::to_value(parsed).unwrap(), value);
        }
        for invalid in [
            json!(null),
            json!(42),
            json!([-1]),
            json!([4294967296_u64]),
            json!([1.5]),
            json!([1, "mixed"]),
            json!(["text", {"role": "user", "content": "mixed"}]),
            json!([[{"role": "user", "content": "nested"}]]),
            json!({"messages": [{"role": "user", "content": "wrapped"}]}),
            json!([{"role": "tool", "content": "unknown role"}]),
            json!([{"role": "user", "content": {"image_url": "not a part list"}}]),
            json!([{"role": "user", "content": [], "extra": "must not disappear"}]),
        ] {
            assert!(serde_json::from_value::<EmbeddingInput>(invalid).is_err());
        }
        assert!(
            matches!(serde_json::from_value::<EmbeddingInput>(json!([])).unwrap(),
            EmbeddingInput::TextBatch(texts) if texts.is_empty())
        );
    }

    #[test]
    fn preserves_native_roles_and_interleaved_media() {
        let value = json!({"messages": [
            {"role": "system", "content": "Inspect the conversation."},
            {"role": "user", "content": [
                {"type": "text", "text": "Before"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"}},
                {"type": "text", "text": "After"},
                {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
                {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
            ]},
            {"role": "assistant", "content": "Previous response"}
        ]});
        let input: ModelInput = serde_json::from_value(value.clone()).unwrap();
        assert!(matches!(input, ModelInput::Messages(_)));
        assert_eq!(serde_json::to_value(input).unwrap(), value);
    }

    #[test]
    fn invalid_messages_cannot_fall_back_to_text() {
        for value in [
            json!(null),
            json!({"ticket": "legacy object"}),
            json!([{ "role": "user", "content": "bare array" }]),
            json!({"messages": [], "metadata": "must not be lost"}),
            json!({"messages": [{"role": "unknown", "content": "x"}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "unknown", "text": "x"}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}),
            json!({"messages": [{"role": "assistant", "content": "x", "tool_calls": []}]}),
        ] {
            assert!(serde_json::from_value::<ModelInput>(value).is_err());
        }
        assert!(matches!(
            serde_json::from_value::<ModelInput>(json!("plain text")).unwrap(),
            ModelInput::Text(_)
        ));
    }
}
