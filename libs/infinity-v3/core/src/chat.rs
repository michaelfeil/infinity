//! Native text conversation rendering, independent of tokenizer implementation.
use crate::input::{ContentPart, Message, MessageContent, MessageRole};
use basetenkenizer::chat_template::{minijinja, ChatTemplateOptions, ChatTemplateRenderer};
use serde_json::{Map, Value};
use std::{fs, path::Path};

pub struct ChatProcessor {
    renderer: ChatTemplateRenderer,
    special_tokens: Map<String, Value>,
}

impl ChatProcessor {
    /// Prefer the standalone template; otherwise use the config's single/default template.
    pub fn load(root: &Path) -> Result<Option<Self>, String> {
        let config = match fs::read(root.join("tokenizer_config.json")) {
            Ok(bytes) => serde_json::from_slice::<Value>(&bytes).map_err(|e| e.to_string())?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(e) => return Err(e.to_string()),
        };
        let template = match fs::read_to_string(root.join("chat_template.jinja")) {
            Ok(template) => Some(template),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                select_template(&config["chat_template"])?.map(str::to_owned)
            }
            Err(e) => return Err(e.to_string()),
        };
        template
            .map(|template| Self::new(&template, &config))
            .transpose()
    }

    fn new(template: &str, config: &Value) -> Result<Self, String> {
        let mut special_tokens = Map::new();
        if let Some(config) = config.as_object() {
            for (name, token) in config {
                if name.ends_with("_token") {
                    if let Some(content) = token.as_str().or_else(|| token["content"].as_str()) {
                        special_tokens.insert(name.clone(), Value::String(content.to_owned()));
                    }
                }
            }
        }
        Ok(Self {
            renderer: ChatTemplateRenderer::new(template).map_err(|e| e.to_string())?,
            special_tokens,
        })
    }

    pub(crate) fn render(&self, messages: Vec<Message>) -> Result<String, String> {
        if messages.is_empty() {
            return Err("Conversation must contain at least one message".into());
        }
        let mut text_messages = Vec::with_capacity(messages.len());
        for message in messages {
            let text = match message.content {
                MessageContent::Text(text) => text,
                MessageContent::Parts(parts) => {
                    let mut text = String::new();
                    for part in parts {
                        match part {
                            ContentPart::Text { text: part } => text.push_str(&part),
                            _ => return Err("The text conversation processor does not support images, audio, or video".into()),
                        }
                    }
                    text
                }
            };
            text_messages.push(Message {
                role: message.role,
                content: MessageContent::Text(text),
            });
        }
        self.render_native(&text_messages, false)
    }

    /// The model processor validates role/modality support before native rendering.
    pub(crate) fn render_native(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        self.render_with_context(messages, add_generation_prompt, Map::new())
    }

    /// Render a trained decision prompt with reasoning disabled.
    pub fn render_decision(&self, system: &str, user: &str) -> Result<String, String> {
        let messages = [
            Message {
                role: MessageRole::System,
                content: MessageContent::Text(system.into()),
            },
            Message {
                role: MessageRole::User,
                content: MessageContent::Text(user.into()),
            },
        ];
        self.render_with_context(
            &messages,
            true,
            Map::from_iter([("enable_thinking".into(), Value::Bool(false))]),
        )
    }

    fn render_with_context(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        extra_context: Map<String, Value>,
    ) -> Result<String, String> {
        let rendered = self
            .renderer
            .render_value(
                minijinja::Value::from_serialize(messages),
                ChatTemplateOptions {
                    add_generation_prompt,
                    special_tokens: self.special_tokens.clone(),
                    extra_context,
                    ..Default::default()
                },
            )
            .map_err(|_| "The model's chat template rejected this conversation".to_owned())?;
        if rendered.is_empty() {
            return Err("The model's chat template produced an empty conversation".into());
        }
        Ok(rendered)
    }
}

fn select_template(value: &Value) -> Result<Option<&str>, String> {
    if value.is_null() {
        return Ok(None);
    }
    if let Some(template) = value.as_str() {
        return Ok(Some(template));
    }
    let default = if let Some(templates) = value.as_array() {
        templates
            .iter()
            .find(|entry| {
                entry["name"] == "default" || (templates.len() == 1 && entry["name"].is_string())
            })
            .and_then(|entry| entry["template"].as_str())
    } else if let Some(templates) = value.as_object() {
        let selected = if templates.len() == 1 {
            templates.values().next()
        } else {
            templates.get("default")
        };
        selected.and_then(Value::as_str)
    } else {
        None
    };
    default.map(Some).ok_or_else(|| {
        "Expected a single usable chat template or a named `default` template".into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn workers_match_hf_without_duplicate_special_tokens_on_fast_and_fallback_paths() {
        use crate::tokenization::{EncodingInput, Tokenization};
        use tokenizers::{AddedToken, Tokenizer, TruncationDirection};
        let wordpiece = tokenizers::models::wordpiece::WordPiece::builder()
            .vocab(
                [
                    ("[UNK]".to_owned(), 0),
                    ("hello".to_owned(), 1),
                    ("world".to_owned(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .build()
            .unwrap();
        let mut fallback = Tokenizer::new(wordpiece);
        fallback.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        for mut hf in [crate::fast_tokenization::test_tokenizer(), fallback] {
            hf.add_special_tokens(&[
                AddedToken::from("<s>", true),
                AddedToken::from("</s>", true),
            ]);
            let bos = hf.token_to_id("<s>").unwrap();
            let eos = hf.token_to_id("</s>").unwrap();
            hf.with_post_processor(Some(
                tokenizers::processors::template::TemplateProcessing::builder()
                    .try_single("<s> $A </s>")
                    .unwrap()
                    .special_tokens(vec![("<s>", bos), ("</s>", eos)])
                    .build()
                    .unwrap(),
            ));
            let expected = hf
                .encode("<s>hello world</s>", false)
                .unwrap()
                .get_ids()
                .to_vec();
            let processor = ChatProcessor::new(
                "{{ bos_token }}{% for m in messages %}{{ m.content }}{% if not loop.last %} {% endif %}{% endfor %}{{ eos_token }}",
                &json!({"bos_token":"<s>", "eos_token":{"content":"</s>"}}),
            ).unwrap();
            let ordinary = hf.encode("prefix hello", true).unwrap().get_ids().to_vec();
            let workers =
                Tokenization::new(1, hf, 128, 0, Some("prefix ".into()), None, Some(processor));
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let messages = || {
                    EncodingInput::Messages(
                        serde_json::from_value(json!([
                            {"role":"user", "content":[{"type":"text", "text":"hello"}]},
                            {"role":"assistant", "content":"world"}
                        ]))
                        .unwrap(),
                    )
                };
                let output = workers
                    .encode_embedding(messages(), false, TruncationDirection::Right, None)
                    .await
                    .unwrap();
                assert_eq!(output.input_ids, expected);
                assert_eq!(output.input_ids.iter().filter(|&&id| id == bos).count(), 1);
                assert_eq!(output.input_ids.iter().filter(|&&id| id == eos).count(), 1);
                let output = workers
                    .encode(messages(), false, TruncationDirection::Right, None)
                    .await
                    .unwrap();
                assert_eq!(output.input_ids, expected);
                let (rendered, output) = workers.tokenize(messages(), true, None).await.unwrap();
                assert_eq!(rendered.as_deref(), Some("<s>hello world</s>"));
                assert_eq!(output.get_ids(), expected);
                let output = workers
                    .encode_embedding(
                        "hello".to_owned().into(),
                        false,
                        TruncationDirection::Right,
                        None,
                    )
                    .await
                    .unwrap();
                assert_eq!(output.input_ids, ordinary);
                let error = workers
                    .encode_embedding(
                        messages(),
                        false,
                        TruncationDirection::Right,
                        Some("query".into()),
                    )
                    .await
                    .unwrap_err();
                assert!(error.to_string().contains("cannot be combined"));
            });
        }
    }

    #[test]
    fn renders_ordered_text_with_configured_special_tokens() {
        let processor = ChatProcessor::new(
            "{{ bos_token }}{% for m in messages %}{{ m.role }}:{{ m.content }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
            &json!({"bos_token": {"content": "<bos>"}, "eos_token": "<eos>"}),
        ).unwrap();
        let messages = serde_json::from_value(json!([
            {"role": "system", "content": "instructions"},
            {"role": "developer", "content": "policy"},
            {"role": "user", "content": [{"type":"text", "text":"one"}, {"type":"text", "text":" two"}]},
            {"role": "assistant", "content": "three"}
        ])).unwrap();
        assert_eq!(
            processor.render(messages).unwrap(),
            "<bos>system:instructions<eos>developer:policy<eos>user:one two<eos>assistant:three<eos>"
        );
    }

    #[test]
    fn rejects_unsupported_content_before_template_can_drop_it() {
        let empty = ChatProcessor::new("", &Value::Null).unwrap();
        let messages = serde_json::from_value(json!([{"role":"user", "content":"hello"}])).unwrap();
        assert!(empty
            .render(messages)
            .unwrap_err()
            .contains("empty conversation"));
        let processor = ChatProcessor::new("constant", &Value::Null).unwrap();
        for value in [
            json!([]),
            json!([{"role":"assistant", "content":[{"type":"image_url", "image_url":{"url":"https://example.com/?secret=hidden"}}]}]),
            json!([{"role":"user", "content":[{"type":"input_audio", "input_audio":{"data":"AA==", "format":"wav"}}]}]),
        ] {
            let error = processor
                .render(serde_json::from_value(value).unwrap())
                .unwrap_err();
            assert!(!error.contains("hidden"));
        }
    }

    #[test]
    fn template_owns_text_role_validation() {
        let processor = ChatProcessor::new(
            "{% for m in messages %}{% if m.role != 'user' %}{{ raise_exception('private template detail') }}{% endif %}{{ m.content }}{% endfor %}",
            &Value::Null,
        ).unwrap();
        let messages = serde_json::from_value(json!([
            {"role": "system", "content": "private request detail"},
            {"role": "user", "content": "hello"}
        ]))
        .unwrap();
        assert_eq!(
            processor.render(messages).unwrap_err(),
            "The model's chat template rejected this conversation"
        );
    }

    #[test]
    fn named_templates_select_the_sole_entry_or_explicit_default() {
        assert_eq!(
            select_template(&json!({"default":"chosen", "tools":"other"})).unwrap(),
            Some("chosen")
        );
        assert_eq!(
            select_template(&json!([{"name":"default", "template":"chosen"}])).unwrap(),
            Some("chosen")
        );
        assert_eq!(
            select_template(&json!({"tools":"sole"})).unwrap(),
            Some("sole")
        );
        assert_eq!(
            select_template(&json!([{"name":"chat", "template":"sole"}])).unwrap(),
            Some("sole")
        );
        assert_eq!(
            select_template(&json!([
                {"name":"chat", "template":"other"},
                {"name":"default", "template":"chosen"}
            ]))
            .unwrap(),
            Some("chosen")
        );
        for ambiguous in [
            json!({"tools":"one", "chat":"two"}),
            json!([{"name":"tools", "template":"one"}, {"name":"chat", "template":"two"}]),
            json!([]),
            json!({}),
            json!([{"template":"missing name"}]),
        ] {
            assert!(select_template(&ambiguous).is_err());
        }
    }
}
