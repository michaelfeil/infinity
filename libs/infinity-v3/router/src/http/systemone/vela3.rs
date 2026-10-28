//! Core Vela encoder decisions. Task adapters are deliberately not substituted by the base head.
use super::*;
use infinity_v3_backend::ClefField;
use infinity_v3_core::multimodal::{MultimodalConfig, UltraVbertProcessor};

pub(crate) struct Encoder {
    tokenizer: Tokenizer,
    markers: Option<[u32; 4]>,
    bos: u32,
    eos: u32,
    sep: u32,
    replacement: u32,
    limit: usize,
    images: Option<UltraVbertProcessor>,
    pub(super) policy: Option<crate::vela3::Policy>,
}

impl Encoder {
    pub(super) fn load(
        root: &Path,
        mut tokenizer: Tokenizer,
        limit: usize,
        media: MultimodalConfig,
    ) -> anyhow::Result<Self> {
        let config: Value = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        let vision = config["model_type"] == "ultravbert";
        anyhow::ensure!(
            vision || config["model_type"] == "ultrabert",
            "Vela3 encoder requires UltraBERT or UltraVBERT"
        );
        let token = |name| {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| anyhow::anyhow!("Vela tokenizer missing {name}"))
        };
        let bos = token("<|endoftext|>")?;
        let sep = token("<|sep|>")?;
        let replacement = token("?")?;
        let markers = if vision {
            None
        } else {
            let mut ids = [0; 4];
            for (i, name) in ["[Q]", "[O]", "[SEP_SCHEMA]", "[SEG_user]"]
                .iter()
                .enumerate()
            {
                ids[i] = config["decision"]["marker_ids"][name]
                    .as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| anyhow::anyhow!("Missing Vela marker {name}"))?;
            }
            anyhow::ensure!(
                config["decision"]["n_vocab_base"] == tokenizer.get_vocab_size(true),
                "Vela tokenizer vocabulary differs from checkpoint"
            );
            Some(ids)
        };
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let images = if vision {
            Some(UltraVbertProcessor::load(root, tokenizer.clone(), media)?)
        } else {
            None
        };
        Ok(Self {
            tokenizer,
            markers,
            bos,
            eos: bos,
            sep,
            replacement,
            limit: limit.min(32768),
            images,
            policy: None,
        })
    }

    fn tokens(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        if self.markers.is_none() {
            for id in &mut ids {
                if *id >= 49152 {
                    *id = self.replacement;
                }
            }
        }
        Ok(ids)
    }

    pub(super) fn prepare(&self, request: SystemOneRequest) -> Result<Vec<Question>, String> {
        if let Some(policy) = &self.policy {
            policy.validate(&request.questions)?;
        }
        if request.head_max_len.is_some() {
            return Err("head_max_len is specific to Laya".into());
        }
        if request.questions.is_empty() {
            return Err("Vela requires at least one question".into());
        }
        let ModelInput::Text(state) = request.state else {
            return Err("Vela state must be normalized before prompting".into());
        };
        let limit = request.max_len.unwrap_or(self.limit).min(self.limit);
        request.questions.into_iter().map(|(id, spec)| {
            if id.is_empty() { return Err("Question IDs must be nonempty".into()); }
            let spec = spec.as_object().ok_or("Question must be an object")?;
            if spec.keys().any(|key| !matches!(key.as_str(), "type" | "instructions" | "criteria")) {
                return Err("Core Vela questions support only type, instructions and criteria".into());
            }
            let kind = spec.get("type").and_then(Value::as_str).ok_or("Question needs type")?;
            let criteria = spec.get("criteria").cloned().unwrap_or(Value::Null);
            let options: Vec<(String,String)> = match kind {
                "choice" => match &criteria {
                    Value::Object(map) => map.iter().map(|(name, desc)| {
                        let rendered = if self.markers.is_some() { text(desc) } else { python_repr(desc) };
                        let described = if self.markers.is_some() { !desc.is_null() && !rendered.is_empty() } else { d3::truthy(desc) };
                        (name.clone(), if described { format!("{name}: {rendered}") } else { name.clone() })
                    }).collect(),
                    Value::Array(values) if self.markers.is_none() && values.iter().all(Value::is_string) => values.iter().map(|v| { let s = text(v); (s.clone(),s) }).collect(),
                    _ => return Err("Vela choice criteria must be an object".into()),
                },
                "noul" => [("no", if self.markers.is_some() { "no" } else { "No / false" }),
                           ("yes", if self.markers.is_some() { "yes" } else { "Yes / true" })]
                    .into_iter().map(|(a,b)|(a.into(),b.into())).collect(),
                _ => return Err("Core Vela encoder supports choice and noul; task adapters and spans are not enabled".into()),
            };
            if options.is_empty() || options.len() > 255 || options.iter().any(|(k,_)| k.is_empty()) ||
                options.iter().map(|(k,_)|k).collect::<HashSet<_>>().len() != options.len() {
                return Err("Vela choice needs 1-255 distinct, nonempty option keys".into());
            }
            let instructions = spec.get("instructions").filter(|v|self.markers.is_some() || d3::truthy(v)).map(text).unwrap_or_default();
            let mut ids = vec![self.bos];
            let input = if let Some([q,o,schema,user]) = self.markers {
                ids.push(q); ids.extend(self.tokens(&instructions)?);
                let mut markers = vec![1,0,0];
                for (_, desc) in &options { markers.push(ids.len()); ids.push(o); ids.extend(self.tokens(desc)?); }
                ids.push(schema); markers[1] = ids.len(); ids.push(user); ids.extend(self.tokens(&state)?);
                markers[2] = ids.len(); ids.push(self.eos);
                DecisionInput::Laya { question_type: 0, markers }
            } else {
                if !state.trim().is_empty() { ids.extend(self.tokens(&format!("Input: {}",state.trim()))?); }
                ids.extend(self.tokens(&format!(" Question: {}",instructions.trim()))?);
                let mut spans = Vec::new();
                for (_, desc) in &options {
                    ids.push(self.sep); let start = ids.len(); ids.extend(self.tokens(&format!(" {desc}"))?);
                    if start == ids.len() { return Err("Vela option must contain tokens".into()); }
                    spans.push((start, ids.len()));
                }
                ids.push(self.sep);
                DecisionInput::Clef { fields: vec![ClefField { kind:0, question:(0,1), options:spans }] }
            };
            if ids.len() > limit { return Err(format!("Vela prompt has {} tokens, exceeding max_len {limit}; nothing is truncated", ids.len())); }
            let n = ids.len();
            Ok(Question { id, kind: kind.into(), labels:options.into_iter().map(|(k,_)|k).collect(), criteria,
                compute_chars: state.chars().count() + instructions.chars().count(), input,
                encoding:ValidEncoding { input_ids:ids, token_type_ids:vec![0;n], position_ids:(0..n as u32).collect(), tokens:vec![], offsets:vec![],multimodal:None } })
        }).collect()
    }

    pub(super) async fn attach_images(
        &self,
        questions: &mut [Question],
        sources: Vec<String>,
        videos: Vec<String>,
        max_len: Option<usize>,
    ) -> Result<Result<(), String>, infinity_v3_core::TextEmbeddingsError> {
        if !videos.is_empty() {
            return Ok(Err("Vela encoders do not support video".into()));
        }
        let Some(processor) = &self.images else {
            return Ok(Err("vela3-fast is text-only".into()));
        };
        let media = processor.prepare(sources).await?;
        for question in questions {
            let delta = media.prefix.len().saturating_sub(1);
            if question.encoding.input_ids.len().saturating_add(delta)
                > max_len.unwrap_or(self.limit).min(self.limit)
            {
                return Ok(Err(
                    "Vela image prompt exceeds max_len; nothing is truncated".into(),
                ));
            }
            question
                .encoding
                .input_ids
                .splice(..1, media.prefix.iter().copied());
            if let DecisionInput::Clef { fields } = &mut question.input {
                for field in fields {
                    for (a, b) in &mut field.options {
                        *a += delta;
                        *b += delta;
                    }
                }
            }
            let n = question.encoding.input_ids.len();
            question.encoding.token_type_ids = vec![0; n];
            question.encoding.position_ids = (0..n as u32).collect();
            question.encoding.multimodal =
                Some(Arc::new(infinity_v3_backend::MultimodalEncoding {
                    images: media.images.clone(),
                    audios: vec![],
                    position_ids: std::array::from_fn(|_| question.encoding.position_ids.clone()),
                    memory: Some(media.memory.clone()),
                }));
        }
        Ok(Ok(()))
    }
}

#[derive(Clone, Copy)]
enum Syntax {
    Json,
    Python,
}

pub(super) fn text(value: &Value) -> String {
    render(value, Syntax::Json, false)
}
fn python_repr(value: &Value) -> String {
    render(value, Syntax::Python, false)
}

// Both encoders render top-level strings verbatim. Containers differ in
// quoting, separators and scalar spelling, but share the same traversal.
fn render(value: &Value, syntax: Syntax, nested: bool) -> String {
    let python = matches!(syntax, Syntax::Python);
    match value {
        Value::String(s) if !nested => s.clone(),
        Value::String(s) if python => python_string(s),
        Value::Null if python => "None".into(),
        Value::Bool(v) if python => if *v { "True" } else { "False" }.into(),
        Value::Number(n) => python_number(n),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(|value| render(value, syntax, true))
                .collect::<Vec<_>>()
                .join(if python { ", " } else { "," })
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}{}{}",
                    render(&Value::String(key.clone()), syntax, true),
                    if python { ": " } else { ":" },
                    render(value, syntax, true)
                ))
                .collect::<Vec<_>>()
                .join(if python { ", " } else { "," })
        ),
        _ => value.to_string(),
    }
}
fn python_number(n: &serde_json::Number) -> String {
    if let Some(number) = n.as_f64().filter(|_| n.is_f64()) {
        let rendered = format!("{number:?}");
        if let Some((mantissa, exponent)) = rendered.split_once('e') {
            if let Ok(exponent) = exponent.parse::<i32>() {
                return format!("{mantissa}e{exponent:+03}");
            }
        }
        rendered
    } else {
        n.to_string()
    }
}
fn python_string(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            '\'' | '"' => out.push(c),
            // Prefixing a printable character keeps printable combining marks
            // from being escaped merely because they start a debug string.
            c if format!("a{c}").escape_debug().nth(1) == Some('\\') => {
                let n = c as u32;
                out.push_str(&if n <= 0xff {
                    format!("\\x{n:02x}")
                } else if n <= 0xffff {
                    format!("\\u{n:04x}")
                } else {
                    format!("\\U{n:08x}")
                });
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

pub(super) fn answer(
    question: &Question,
    output: DecisionOutput,
) -> Result<ResultValue, AnswerError> {
    if output.logits.len() != question.labels.len()
        || output.logits.is_empty()
        || output.logits.iter().any(|x| !x.is_finite())
    {
        return Err("Invalid Vela option logits".into());
    }
    let max = output
        .logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = output
        .logits
        .iter()
        .map(|&x| (f64::from(x) - f64::from(max)).exp())
        .sum();
    let probability = |x: f32| (f64::from(x) - f64::from(max)).exp() / sum;
    if question.kind == "noul" {
        return ResultValue::fields([
            ("type", ResultValue::string("noul")?),
            (
                "noul",
                ResultValue::number(probability(
                    *output.logits.get(1).ok_or("Missing yes option")?,
                )),
            ),
        ]);
    }
    let mut probabilities = ResultValue::object(question.labels.len())?;
    let mut best = (0, f32::NEG_INFINITY);
    for (i, (label, &logit)) in question.labels.iter().zip(&output.logits).enumerate() {
        if logit > best.1 {
            best = (i, logit);
        }
        probabilities.put(label, ResultValue::number(probability(logit)))?;
    }
    ResultValue::fields([
        ("type", ResultValue::string("choice")?),
        (
            "choice",
            ResultValue::string(question.labels.get(best.0).ok_or("Missing chosen option")?)?,
        ),
        ("probabilities", probabilities),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires upstream-generated checkpoint fixtures in VELA3_REFERENCE_DIR"]
    fn checkpoint_encoder_prompts_match_reference() -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(std::env::var("VELA3_REFERENCE_DIR")?);
        for name in ["vela3-fast", "vela3-fast-vision"] {
            let path = root.join(name);
            let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let encoder = Encoder::load(&path, tokenizer, 32768, Default::default())?;
            let references: Value = serde_json::from_slice(&std::fs::read(
                root.join(format!("{name}-reference.json")),
            )?)?;
            for (case, reference) in references
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("Invalid reference"))?
            {
                if reference["request"].get("images").is_some() {
                    continue;
                }
                let request: d3::Request = serde_json::from_value(reference["request"].clone())?;
                let (request, _, _) = request.into_vela_common().map_err(anyhow::Error::msg)?;
                let questions = encoder.prepare(request).map_err(anyhow::Error::msg)?;
                for question in questions {
                    let expected: Vec<u32> =
                        serde_json::from_value(reference["records"][&question.id]["ids"].clone())?;
                    assert_eq!(
                        question.encoding.input_ids, expected,
                        "{name} {case} {}",
                        question.id
                    );
                    if let DecisionInput::Laya { markers, .. } = question.input {
                        let rec = &reference["records"][&question.id];
                        let mut expected = vec![
                            rec["q_pos"].as_u64().unwrap() as usize,
                            rec["pool"][0].as_u64().unwrap() as usize,
                            rec["pool"][1].as_u64().unwrap() as usize,
                        ];
                        expected.extend(serde_json::from_value::<Vec<usize>>(
                            rec["opt_pos"].clone(),
                        )?);
                        assert_eq!(markers, expected);
                    }
                }
            }
        }
        Ok(())
    }
}
