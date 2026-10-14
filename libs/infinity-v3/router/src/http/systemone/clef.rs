//! Exact segmented prompt encoding for Cloudflare's joint schema head.
use super::super::result_value::{AnswerError, ResultValue};
use super::{Question, SystemOneRequest};
use infinity_v3_backend::{ClefField, DecisionInput, DecisionOutput};
use infinity_v3_core::{
    input::{ContentPart, MessageContent, MessageRole, ModelInput},
    tokenization::ValidEncoding,
};
use serde_json::{json, Map, Value};
use tokenizers::Tokenizer;

pub struct Clef {
    tokenizer: Tokenizer,
    max_input_length: usize,
}
impl Clef {
    pub fn load(mut tokenizer: Tokenizer, max_input_length: usize) -> anyhow::Result<Self> {
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        anyhow::ensure!(
            tokenizer.token_to_id("<|im_start|>").is_some(),
            "Clef requires a Qwen tokenizer"
        );
        Ok(Self {
            tokenizer,
            max_input_length,
        })
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }
    fn append(&self, ids: &mut Vec<u32>, text: &str) -> Result<(), String> {
        ids.extend(self.encode(text)?);
        Ok(())
    }
    pub(super) fn prepare(&self, request: SystemOneRequest) -> Result<Vec<Question>, String> {
        if request.head_max_len.is_some() {
            return Err("Clef does not use head_max_len".into());
        }
        let max_len = request.max_len.unwrap_or(self.max_input_length);
        if max_len == 0 || max_len > self.max_input_length {
            return Err(format!(
                "max_len must be between 1 and {}",
                self.max_input_length
            ));
        }
        if request.questions.is_empty() {
            return Err("At least one question is required".into());
        }
        let state = match request.state {
            ModelInput::Text(s) => s,
            ModelInput::Messages(m) => {
                if m.messages.is_empty() {
                    return Err("messages must not be empty".into());
                }
                for message in &m.messages {
                    if !matches!(message.role, MessageRole::User | MessageRole::Assistant) {
                        return Err("Clef state messages must use user or assistant roles".into());
                    }
                    if let MessageContent::Parts(parts) = &message.content {
                        if parts.iter().any(|p| !matches!(p, ContentPart::Text { .. })) {
                            return Err(
                                "Clef image/audio/video inputs are not yet supported".into()
                            );
                        }
                    }
                }
                render(&serde_json::to_value(&m).map_err(|e| e.to_string())?)
            }
        };
        if state.chars().count() > 50_000 {
            return Err("state exceeds 50000 characters".into());
        }
        let safe = |text: &str| -> Result<(), String> {
            if text.contains("<|") || text.contains("<think>") || text.contains("</think>") {
                Err("Input must not contain reserved chat tokens".into())
            } else {
                Ok(())
            }
        };
        safe(&state)?;
        let mut ids=self.encode("<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n")?;
        self.append(&mut ids, &state)?;
        self.append(&mut ids, "\n\nSCHEMA FIELDS:\n")?;
        let mut fields = Vec::new();
        let mut descriptors = Vec::new();
        let mut total = 0;
        for (i, (id, q)) in request.questions.into_iter().enumerate() {
            let obj = q.as_object().ok_or("Each question must be an object")?;
            if obj
                .keys()
                .any(|k| !matches!(k.as_str(), "type" | "instructions" | "criteria"))
            {
                return Err(format!("Question {id}: unsupported field"));
            }
            let kind = q["type"].as_str().ok_or("Question type is required")?;
            let type_id = match kind {
                "noul" => 0,
                "choice" => 1,
                "score" => 2,
                _ => return Err("Question type must be choice, score, or noul".into()),
            };
            safe(&id)?;
            self.append(
                &mut ids,
                &format!("\nFIELD {}\nID: {id}\nTYPE: {kind}\nINSTRUCTION: ", i + 1),
            )?;
            let instructions = match q.get("instructions") {
                None | Some(Value::Null) => id.clone(),
                Some(Value::String(s)) if s.is_empty() => id.clone(),
                Some(v) => render(v),
            };
            safe(&instructions)?;
            let start = ids.len();
            self.append(&mut ids, &instructions)?;
            let question = (start, ids.len());
            if question.0 == question.1 {
                return Err("Question instruction must not tokenize to an empty span".into());
            }
            self.append(&mut ids, "\nALLOWED OPTIONS:\n")?;
            let criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
            let options: Vec<(String, Value)> = match kind {
                "noul" => {
                    if !criteria.is_null()
                        && (!criteria.is_object()
                            || criteria
                                .as_object()
                                .unwrap()
                                .keys()
                                .any(|k| k != "true" && k != "false"))
                    {
                        return Err("noul criteria must contain only true and false".into());
                    }
                    [
                        ("true", "The proposition is true or the answer is yes."),
                        ("false", "The proposition is false or the answer is no."),
                    ]
                    .into_iter()
                    .map(|(k, default)| {
                        (
                            k.to_string(),
                            criteria.get(k).cloned().unwrap_or(json!(default)),
                        )
                    })
                    .collect()
                }
                "choice" => {
                    let map = criteria
                        .as_object()
                        .ok_or("choice criteria must be an object")?;
                    let mut v: Vec<_> = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                    v.sort_by(|a, b| a.0.cmp(&b.0));
                    v
                }
                _ => criteria
                    .as_array()
                    .ok_or("score criteria must be an array")?
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), v.clone()))
                    .collect(),
            };
            total += options.len();
            if options.is_empty() || total > 512 {
                return Err(
                    "At least one option per question and at most 512 total options are supported"
                        .into(),
                );
            }
            let mut spans = Vec::new();
            let mut labels = Vec::new();
            for (j, (label, description)) in options.into_iter().enumerate() {
                self.append(&mut ids, &format!("OPTION {}: ", j + 1))?;
                let mut semantics = Map::new();
                semantics.insert("option_id".into(), json!(label));
                if !description.is_null() {
                    semantics.insert("description".into(), description);
                }
                let text = render(&Value::Object(semantics));
                safe(&text)?;
                let start = ids.len();
                self.append(&mut ids, &text)?;
                spans.push((start, ids.len()));
                labels.push(label);
                self.append(&mut ids, "\n")?;
            }
            self.append(&mut ids, "END FIELD\n")?;
            fields.push(ClefField {
                kind: type_id,
                question,
                options: spans,
            });
            descriptors.push(json!({"id":id,"type":kind,"labels":labels,"criteria":criteria}));
        }
        self.append(
            &mut ids,
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
        )?;
        // Preserve all schema spans; reject rather than truncate the state silently.
        if ids.len() > max_len {
            return Err(format!(
                "Clef prompt requires {} tokens; maximum is {max_len}",
                ids.len()
            ));
        }
        let n = ids.len();
        Ok(vec![Question {
            id: String::new(),
            kind: "joint".into(),
            labels: vec![],
            criteria: Value::Array(descriptors),
            compute_chars: self
                .tokenizer
                .decode(&ids, false)
                .map_err(|e| e.to_string())?
                .chars()
                .count(),
            encoding: ValidEncoding {
                input_ids: ids,
                token_type_ids: vec![0; n],
                position_ids: (0..n as u32).collect(),
                tokens: vec![],
                offsets: vec![],
                multimodal: None,
            },
            input: DecisionInput::Clef { fields },
        }])
    }
}
// Match json.dumps(..., ensure_ascii=False, separators=(",", ":"), sort_keys=True).
fn render(v: &Value) -> String {
    fn sorted(v: &Value) -> Value {
        match v {
            Value::Object(m) => {
                let mut keys: Vec<_> = m.keys().collect();
                keys.sort();
                Value::Object(
                    keys.into_iter()
                        .map(|k| (k.clone(), sorted(&m[k])))
                        .collect(),
                )
            }
            Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
            _ => v.clone(),
        }
    }
    match v {
        Value::String(s) => s.clone(),
        _ => sorted(v).to_string(),
    }
}
pub(super) fn answer(
    question: &Question,
    output: DecisionOutput,
) -> Result<ResultValue, AnswerError> {
    let mut answers = ResultValue::object(
        question
            .criteria
            .as_array()
            .ok_or("Invalid Clef descriptors")?
            .len(),
    )?;
    let mut offset = 0;
    for descriptor in question
        .criteria
        .as_array()
        .ok_or("Invalid Clef descriptors")?
    {
        let labels = descriptor["labels"]
            .as_array()
            .ok_or("Invalid Clef labels")?;
        let n = labels.len();
        let logits = output
            .logits
            .get(offset..offset + n)
            .ok_or("Invalid Clef logits")?;
        offset += n;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite Clef logits".into());
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut p = super::super::result_storage::vector(logits.len())?;
        p.extend(logits.iter().map(|&v| (v as f64 - max).exp()));
        let sum: f64 = p.iter().sum();
        for v in &mut p {
            *v /= sum;
        }
        let best = p
            .iter()
            .enumerate()
            .fold(0, |best, (i, &v)| if v > p[best] { i } else { best });
        let mut probabilities = ResultValue::object(labels.len())?;
        for (label, p) in labels.iter().zip(&p) {
            probabilities.put(
                label.as_str().unwrap(),
                ResultValue::number(super::round(*p)),
            )?;
        }
        let answer = match descriptor["type"].as_str().unwrap() {
            "noul" => ResultValue::fields([
                ("type", ResultValue::string("noul")?),
                ("noul", ResultValue::number(super::round(p[0]))),
            ])?,
            "choice" => ResultValue::fields([
                ("type", ResultValue::string("choice")?),
                ("choice", ResultValue::copy(&labels[best])?),
                ("confidence", ResultValue::number(super::round(p[best]))),
                ("probabilities", probabilities),
            ])?,
            _ => ResultValue::fields([
                ("type", ResultValue::string("score")?),
                (
                    "score",
                    ResultValue::number(super::round(
                        p.iter().enumerate().map(|(i, p)| i as f64 * p).sum(),
                    )),
                ),
                ("confidence", ResultValue::number(super::round(p[best]))),
                ("probabilities", probabilities),
                (
                    "legend",
                    ResultValue::legend(descriptor["criteria"].as_array().unwrap())?,
                ),
            ])?,
        };
        answers.put(descriptor["id"].as_str().unwrap(), answer)?;
    }
    if offset != output.logits.len() {
        return Err("Unexpected Clef logit count".into());
    }
    Ok(answers)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a Clef checkpoint and upstream reference fixture"]
    fn checkpoint_matches_upstream_prompts() {
        let path = std::env::var("CLEF_CHECKPOINT_DIR").unwrap();
        let reference: Value = serde_json::from_slice(
            &std::fs::read(std::env::var("CLEF_REFERENCE_FILE").unwrap()).unwrap(),
        )
        .unwrap();
        let service = Clef::load(
            Tokenizer::from_file(format!("{path}/tokenizer.json")).unwrap(),
            8192,
        )
        .unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let request: SystemOneRequest =
                serde_json::from_value(case["request"].clone()).unwrap();
            let mut prepared = service.prepare(request).unwrap();
            assert_eq!(prepared.len(), 1);
            let q = prepared.pop().unwrap();
            assert_eq!(
                q.compute_chars,
                case["compute_chars"].as_u64().unwrap() as usize
            );
            assert_eq!(
                q.encoding.input_ids,
                serde_json::from_value::<Vec<u32>>(case["input_ids"].clone()).unwrap()
            );
            let DecisionInput::Clef { fields } = &q.input else {
                panic!()
            };
            for (field, expected) in fields.iter().zip(case["fields"].as_array().unwrap()) {
                assert_eq!(field.kind, expected["kind"].as_u64().unwrap() as usize);
                assert_eq!(
                    field.question,
                    serde_json::from_value::<(usize, usize)>(expected["question"].clone()).unwrap()
                );
                assert_eq!(
                    field.options,
                    serde_json::from_value::<Vec<(usize, usize)>>(expected["options"].clone())
                        .unwrap()
                );
            }
            let logits: Vec<Vec<f32>> = serde_json::from_value(case["logits"].clone()).unwrap();
            let actual = answer(
                &q,
                DecisionOutput {
                    memory: None,
                    logits: logits.into_iter().flatten().collect(),
                    action_probability: 1.0,
                },
            )
            .unwrap();
            assert_eq!(actual, case["answers"]);
        }
        let first = reference["cases"][0]["request"].clone();
        let mut request: SystemOneRequest = serde_json::from_value(first.clone()).unwrap();
        request.max_len = Some(1);
        assert!(service.prepare(request).is_err());
        let mut request: SystemOneRequest = serde_json::from_value(first).unwrap();
        request.state=serde_json::from_value(json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]})).unwrap();
        assert!(service.prepare(request).is_err());
    }
}
