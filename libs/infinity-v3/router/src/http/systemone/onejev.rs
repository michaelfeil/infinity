//! OneJev's qev-labels-v2 text/image protocol, using the shared decision queue.
use super::super::result_value::{AnswerError, ResultValue};
use super::*;
use infinity_v3_core::{
    chat::ChatProcessor,
    input::{ContentPart, ImageDetail, MessageContent, MessageRole},
    multimodal::{image_positions, PreparedQwenImages, Qwen3VlProcessor},
};

const SYSTEM: &str = "Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.";
const SPLIT_LABELS: &[&str] = &[
    "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ",
    "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ",
];

pub struct Onejev {
    tokenizer: Tokenizer,
    chat: ChatProcessor,
    codes: Vec<String>,
    token_ids: Vec<u32>,
    max_input_length: usize,
    images: Option<Qwen3VlProcessor>,
}

impl Onejev {
    pub(super) fn load(
        path: &Path,
        mut tokenizer: Tokenizer,
        max_input_length: usize,
    ) -> anyhow::Result<Self> {
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let chat = ChatProcessor::load(path)
            .map_err(anyhow::Error::msg)?
            .ok_or_else(|| anyhow::anyhow!("OneJev requires the checkpoint's chat template"))?;
        let mut codes = (b'A'..=b'Z')
            .map(|c| (c as char).to_string())
            .chain(
                (b'A'..=b'Z')
                    .flat_map(|a| (b'A'..=b'Z').map(move |b| format!("{}{}", a as char, b as char)))
                    .filter(|s| !SPLIT_LABELS.contains(&s.as_str())),
            )
            .take(256)
            .collect::<Vec<_>>();
        let ids = codes
            .iter()
            .map(|code| {
                let encoded = tokenizer
                    .encode(code.as_str(), false)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(
                    if encoded.len() == 1
                        && tokenizer
                            .decode(encoded.get_ids(), false)
                            .map_err(|e| anyhow::anyhow!(e.to_string()))?
                            == *code
                    {
                        Some(encoded.get_ids()[0])
                    } else {
                        None
                    },
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        anyhow::ensure!(
            ids[..26].iter().all(Option::is_some),
            "OneJev requires single-token A-Z answer labels"
        );
        if ids.iter().any(Option::is_none) {
            codes.truncate(26);
        }
        let token_ids = ids
            .into_iter()
            .take(codes.len())
            .map(Option::unwrap)
            .collect();
        Ok(Self {
            tokenizer,
            chat,
            codes,
            token_ids,
            max_input_length,
            images: None,
        })
    }

    pub(super) fn with_images(
        mut self,
        path: &Path,
        config: infinity_v3_core::multimodal::MultimodalConfig,
    ) -> anyhow::Result<Self> {
        let model: Value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        if model["vision_config"].is_object()
            && (path.join("processor_config.json").exists()
                || path.join("preprocessor_config.json").exists())
        {
            self.images = Some(
                Qwen3VlProcessor::load(path, self.tokenizer.clone(), self.max_input_length, config)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?,
            );
        }
        Ok(self)
    }

    pub(super) fn image_sources(request: &mut SystemOneRequest) -> Result<Vec<String>, String> {
        let check_text = |text: &str| {
            if text.contains("<image:") || text.contains("<video:") {
                Err("State text must not contain reserved media placeholders".to_string())
            } else {
                Ok(())
            }
        };
        let ModelInput::Messages(state) = &mut request.state else {
            if let ModelInput::Text(text) = &request.state {
                check_text(text)?;
            }
            return Ok(vec![]);
        };
        let mut sources = Vec::new();
        for message in &mut state.messages {
            match &mut message.content {
                MessageContent::Text(text) => check_text(text)?,
                MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => check_text(text)?,
                            ContentPart::ImageUrl { image_url }
                                if matches!(message.role, MessageRole::User) =>
                            {
                                if !matches!(image_url.detail, None | Some(ImageDetail::Auto)) {
                                    return Err(
                                        "OneJev image detail must be auto or omitted".into()
                                    );
                                }
                                if sources.len() >= 4 {
                                    return Err(
                                        "OneJev supports at most four images per request".into()
                                    );
                                }
                                let placeholder = format!("<image:{}>", sources.len() + 1);
                                sources.push(std::mem::replace(&mut image_url.url, placeholder));
                            }
                            _ => return Err("OneJev supports user images and text only".into()),
                        }
                    }
                }
            }
        }
        Ok(sources)
    }

    pub(super) async fn prepare_images(
        &self,
        sources: Vec<String>,
    ) -> Result<PreparedQwenImages, infinity_v3_core::TextEmbeddingsError> {
        self.images
            .as_ref()
            .ok_or_else(|| {
                infinity_v3_core::TextEmbeddingsError::Validation(
                    "OneJev image processor unavailable".into(),
                )
            })?
            .prepare_images(sources, 1024)
            .await
    }

    pub(super) fn attach_images(
        &self,
        questions: &mut [Question],
        images: PreparedQwenImages,
        max_len: Option<usize>,
    ) -> Result<(), String> {
        let image_id = self
            .tokenizer
            .token_to_id("<|image_pad|>")
            .ok_or("Missing image token")?;
        let limit = max_len
            .unwrap_or(self.max_input_length)
            .min(self.max_input_length);
        for question in questions {
            let encoding = &mut question.encoding;
            let mut ids = Vec::new();
            let mut spans = Vec::new();
            let mut patches = images.images.iter();
            for &id in &encoding.input_ids {
                if id == image_id {
                    let image = patches.next().ok_or("Too many image markers")?;
                    spans.push((ids.len(), image.clone()));
                    ids.resize(ids.len() + image.token_count(), image_id);
                } else {
                    ids.push(id);
                }
            }
            if patches.next().is_some() {
                return Err("Image marker count mismatch".into());
            }
            if ids.len() > limit {
                return Err(
                    "OneJev image prompt exceeds token limit; image spans cannot be truncated"
                        .into(),
                );
            }
            let positions = image_positions(
                ids.len(),
                &spans
                    .iter()
                    .map(|(start, image)| (*start, image.as_ref()))
                    .collect::<Vec<_>>(),
            )?;
            encoding.token_type_ids = vec![0; ids.len()];
            encoding.position_ids = positions[0].clone();
            encoding.input_ids = ids;
            encoding.multimodal = Some(Arc::new(infinity_v3_backend::MultimodalEncoding {
                audios: vec![],
                images: spans,
                position_ids: positions,
                memory: Some(images.memory.clone()),
            }));
        }
        Ok(())
    }

    pub(super) fn prepare(&self, request: SystemOneRequest) -> Result<Vec<Question>, String> {
        let mut image_count = 0;
        let state = match request.state {
            ModelInput::Text(text) => text,
            ModelInput::Messages(messages) => {
                if messages.messages.is_empty() {
                    return Err("State messages must not be empty".into());
                }
                for message in &messages.messages {
                    if !matches!(message.role, MessageRole::User | MessageRole::Assistant) {
                        return Err("OneJev state accepts user and assistant roles".into());
                    }
                    if let MessageContent::Parts(parts) = &message.content {
                        for part in parts {
                            match part {
                                ContentPart::Text { .. } => {}
                                ContentPart::ImageUrl { image_url }
                                    if image_url.url == format!("<image:{}>", image_count + 1) =>
                                {
                                    image_count += 1;
                                }
                                _ => {
                                    return Err(
                                        "OneJev image sources must be validated before prompting"
                                            .into(),
                                    )
                                }
                            }
                        }
                    }
                }
                serde_json::to_string_pretty(&messages).map_err(|e| e.to_string())?
            }
        };
        if state.chars().count() > 50_000 {
            return Err("State exceeds 50000 characters".into());
        }
        if request.head_max_len.is_some() {
            return Err("head_max_len is specific to Laya".into());
        }
        if request.questions.is_empty() {
            return Err("OneJev requires at least one question".into());
        }
        let limit = request
            .max_len
            .unwrap_or(self.max_input_length)
            .min(self.max_input_length);
        let mut total_options = 0;
        request.questions.into_iter().map(|(id, spec)| {
            let spec = spec.as_object().ok_or("Question must be an object")?;
            if spec.keys().any(|key| !matches!(key.as_str(), "type" | "instructions" | "criteria")) { return Err("OneJev question supports only type, instructions and criteria".into()); }
            let kind = spec.get("type").and_then(Value::as_str).ok_or("Question needs type")?;
            let default = match kind {
                "choice" => "Which option applies to the state?",
                "score" => "Which level describes the state?",
                "noul" => "Is the statement true, or is the answer to the question yes?",
                _ => return Err("Question type must be choice, noul or score".into()),
            };
            let instructions = entry(spec.get("instructions").unwrap_or(&Value::Null))?;
            let instructions = if instructions.is_empty() { default } else { &instructions };
            let criteria = spec.get("criteria").cloned().unwrap_or(Value::Null);
            let (labels, names, descriptions): (Vec<String>, Vec<String>, Vec<String>) = match kind {
                "choice" => {
                    let options = criteria.as_object().ok_or("Choice criteria must be an object")?;
                    let labels = options.keys().cloned().collect::<Vec<_>>();
                    (labels.clone(), labels, options.values().map(entry).collect::<Result<_, _>>()?)
                }
                "score" => {
                    let options = criteria.as_array().ok_or("Score criteria must be an array")?;
                    ((0..options.len()).map(|i| i.to_string()).collect(), (0..options.len()).map(|i| format!("level {i}")).collect(), options.iter().map(entry).collect::<Result<_, _>>()?)
                }
                "noul" => {
                    if !criteria.is_null() && (criteria.as_object().is_none_or(|v| v.len() != 2 || !v.contains_key("true") || !v.contains_key("false"))) { return Err("Noul criteria must contain exactly false and true".into()); }
                    let descriptions = [("true", "the statement is true / the answer is yes"), ("false", "the statement is false / the answer is no")].into_iter().map(|(key, default)| {
                        match criteria.get(key).filter(|v| !v.is_null()) { Some(value) => entry(value), None => Ok(default.into()) }
                    }).collect::<Result<_, _>>()?;
                    (vec!["false".into(), "true".into()], vec!["yes".into(), "no".into()], descriptions)
                }
                _ => unreachable!(),
            };
            let n = labels.len();
            total_options += n;
            if n == 0 || n > self.codes.len().min(255) || (kind == "score" && n > 10) || total_options > 512 { return Err(format!("OneJev supports 1-{} options per question and 512 per request", self.codes.len().min(255))); }
            let options = self.codes.iter().zip(names.iter().zip(&descriptions)).map(|(code, (name, description))| {
                if description.is_empty() { format!("{code}. {name}") } else { format!("{code}. {name}: {description}") }
            }).collect::<Vec<_>>().join("\n");
            let header = if kind == "score" { "Rate the state:" } else { "Question:" };
            let word = if n <= 26 { "letter" } else { "label" };
            let user = format!("<state>\n{state}\n</state>\n\n{header} {instructions}\n\nOptions:\n{options}\n\nAnswer with one {word}: {}.", self.codes[..n].join(", "));
            if ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"].iter().any(|token| user.contains(token)) { return Err("State and questions must not contain reserved chat or media tokens".into()); }
            if instructions.contains("<image:") || options.contains("<image:") || user.contains("<video:") { return Err("Questions must not contain media placeholders".into()); }
            let mut prompt = self.chat.render_decision(SYSTEM, &user)?;
            for i in 1..=image_count {
                prompt = prompt.replace(&format!("<image:{i}>"), "<|vision_start|><|image_pad|><|vision_end|>");
            }
            let ids = self.tokenizer.encode(prompt.as_str(), false).map_err(|e| e.to_string())?.get_ids().to_vec();
            if ids.is_empty() || ids.len() > limit { return Err(format!("OneJev prompt has {} tokens, exceeding max_len {limit}; prompts are never truncated", ids.len())); }
            let length = ids.len();
            Ok(Question { id, kind: kind.into(), labels, criteria, compute_chars: prompt.chars().count(),
                encoding: ValidEncoding { multimodal: None, input_ids: ids, token_type_ids: vec![0; length], position_ids: (0..length as u32).collect(), tokens: vec![], offsets: vec![] },
                input: DecisionInput::OptionTokens { token_ids: self.token_ids[..n].to_vec() } })
        }).collect()
    }
}

fn entry(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.trim().into()),
        Value::Object(_) | Value::Array(_) => {
            serde_json::to_string_pretty(value).map_err(|e| e.to_string())
        }
        _ => Err("Instructions and criteria must be strings, objects, arrays or null".into()),
    }
}

pub(super) fn answer(
    question: &Question,
    mut output: DecisionOutput,
) -> Result<ResultValue, AnswerError> {
    // OneJev trains A=yes/B=no; the shared answer formatter expects false/true.
    if question.kind == "noul" && output.logits.len() == 2 {
        output.logits.swap(0, 1);
    }
    rune::answer(question, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_images_are_ordered_and_unsupported_parts_are_rejected() {
        let value = json!({"state":{"messages":[{"role":"user","content":[
            {"type":"text","text":"Compare."},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,first"}},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,second"}}
        ]}]},"questions":{"q":{"type":"noul"}}});
        let mut request = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(
            Onejev::image_sources(&mut request).unwrap(),
            [
                "data:image/png;base64,first",
                "data:image/png;base64,second"
            ]
        );
        assert!(serde_json::to_string(&request.state)
            .unwrap()
            .contains("<image:2>"));
        for case in 0..3 {
            let mut invalid = value.clone();
            match case {
                0 => invalid["state"]["messages"][0]["role"] = json!("assistant"),
                1 => invalid["state"]["messages"][0]["content"][0]["text"] = json!("<image:1>"),
                _ => {
                    invalid["state"]["messages"][0]["content"][1]["image_url"]["detail"] =
                        json!("high")
                }
            }
            assert!(Onejev::image_sources(&mut serde_json::from_value(invalid).unwrap()).is_err());
        }
    }

    #[test]
    #[ignore = "requires OneJev checkpoint and upstream prompt fixtures"]
    fn checkpoint_matches_upstream_prompts() -> anyhow::Result<()> {
        let path = std::env::var("ONEJEV_CHECKPOINT_DIR")?;
        let fixtures: Vec<Value> =
            serde_json::from_slice(&std::fs::read(std::env::var("ONEJEV_REFERENCE_FILE")?)?)?;
        let tokenizer = Tokenizer::from_file(Path::new(&path).join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let model = Onejev::load(Path::new(&path), tokenizer, 32768)?
            .with_images(Path::new(&path), Default::default())?;
        let runtime = tokio::runtime::Runtime::new()?;
        for case in fixtures {
            let mut request: SystemOneRequest = serde_json::from_value(case["request"].clone())?;
            let sources = Onejev::image_sources(&mut request).map_err(anyhow::Error::msg)?;
            let mut questions = model.prepare(request).map_err(anyhow::Error::msg)?;
            if !sources.is_empty() {
                let images = runtime.block_on(model.prepare_images(sources))?;
                model
                    .attach_images(&mut questions, images, None)
                    .map_err(anyhow::Error::msg)?;
            }
            for reference in case["fixtures"].as_array().unwrap() {
                let question = questions
                    .iter()
                    .find(|q| q.id == reference["id"].as_str().unwrap())
                    .unwrap();
                let expected: Vec<u32> = serde_json::from_value(reference["input_ids"].clone())?;
                assert_eq!(question.encoding.input_ids, expected, "{}", question.id);
                if let Some(expected) = reference.get("position_ids") {
                    assert_eq!(
                        question.encoding.multimodal.as_ref().unwrap().position_ids,
                        serde_json::from_value::<[Vec<u32>; 3]>(expected.clone())?
                    );
                }
                let DecisionInput::OptionTokens { token_ids } = &question.input else {
                    panic!("expected option tokens");
                };
                assert_eq!(
                    *token_ids,
                    serde_json::from_value::<Vec<u32>>(reference["token_ids"].clone())?
                );
                let logits = serde_json::from_value(reference["logits"].clone())?;
                let actual = answer(
                    question,
                    DecisionOutput {
                        memory: None,
                        logits,
                        action_probability: 1.0,
                    },
                )
                .map_err(anyhow::Error::msg)?;
                let expected = &case["answers"][&question.id];
                for field in ["noul", "score", "confidence"] {
                    if let Some(value) = expected[field].as_f64() {
                        assert!((actual[field].as_f64().unwrap() - value).abs() < 1e-6);
                    }
                }
                assert_eq!(actual["choice"], expected["choice"]);
            }
        }
        let media = json!({"state":{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]},"questions":{"q":{"type":"noul"}}});
        assert!(model
            .prepare(serde_json::from_value(media)?)
            .err()
            .unwrap()
            .contains("validated before prompting"));
        let oversized =
            json!({"state":"Invoice is paid.","max_len":1,"questions":{"q":{"type":"noul"}}});
        assert!(model
            .prepare(serde_json::from_value(oversized)?)
            .err()
            .unwrap()
            .contains("never truncated"));
        Ok(())
    }
}
