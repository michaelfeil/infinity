//! Jev typed decisions, submitted to infinity_v3's shared queue and replica pool.
use super::response_format::Json;
use super::result_value::{AnswerError, ResultValue};
use axum::{
    extract::Extension,
    http::{HeaderMap, StatusCode},
};
use infinity_v3_backend::{DecisionInput, DecisionOutput};
use infinity_v3_core::input::ModelInput;
use infinity_v3_core::{infer::Infer, tokenization::ValidEncoding};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct SystemOneRequest {
    pub state: ModelInput,
    pub questions: Map<String, Value>,
    pub model: Option<String>,
    pub max_len: Option<usize>,
    pub head_max_len: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Config {
    max_len: usize,
    head_max_len: usize,
    #[serde(default)]
    temperature: Vec<f32>,
    #[serde(default)]
    temperature_by_options: HashMap<String, f32>,
}

mod clef;
mod d3;
mod onejev;
mod pplx;
mod rune;

pub enum SystemOne {
    Laya(Laya),
    Rune(rune::Rune),
    Onejev(onejev::Onejev),
    Clef(clef::Clef),
    Pplx(pplx::Pplx),
    D3(Box<d3::D3>),
}

impl SystemOne {
    pub fn load(
        path: &Path,
        tokenizer: Tokenizer,
        max_input_length: usize,
        multimodal_config: infinity_v3_core::multimodal::MultimodalConfig,
        protocol: Option<crate::DecisionProtocol>,
    ) -> anyhow::Result<Self> {
        if path.join("rl_agent_config.json").exists() {
            Ok(Self::Laya(Laya::load(path, tokenizer, max_input_length)?))
        } else if matches!(protocol, Some(crate::DecisionProtocol::Onejev)) {
            Ok(Self::Onejev(
                onejev::Onejev::load(path, tokenizer, max_input_length)?
                    .with_images(path, multimodal_config)?,
            ))
        } else if matches!(protocol, Some(crate::DecisionProtocol::D3)) {
            Ok(Self::D3(Box::new(d3::D3::load(
                path,
                tokenizer,
                max_input_length,
                multimodal_config,
            )?)))
        } else if matches!(protocol, Some(crate::DecisionProtocol::Pplx)) {
            Ok(Self::Pplx(pplx::Pplx::load(
                path,
                tokenizer,
                max_input_length,
            )?))
        } else if matches!(protocol, Some(crate::DecisionProtocol::Clef)) {
            Ok(Self::Clef(clef::Clef::load(tokenizer, max_input_length)?))
        } else {
            Ok(Self::Rune(
                rune::Rune::load(tokenizer, max_input_length)?
                    .with_images(path, multimodal_config)?,
            ))
        }
    }

    fn prepare(&self, request: SystemOneRequest) -> Result<Vec<Question>, String> {
        match self {
            Self::Laya(model) => model.prepare(request),
            Self::Rune(model) => model.prepare(request),
            Self::Onejev(model) => model.prepare(request),
            Self::Clef(model) => model.prepare(request),
            Self::Pplx(model) => model.prepare(request),
            Self::D3(model) => model.prepare(request),
        }
    }

    fn answer(
        &self,
        question: &Question,
        output: DecisionOutput,
    ) -> Result<ResultValue, AnswerError> {
        match self {
            Self::Laya(model) => model.answer(question, output),
            Self::Rune(_) => rune::answer(question, output),
            Self::Onejev(_) => onejev::answer(question, output),
            Self::Clef(_) => clef::answer(question, output),
            Self::Pplx(model) => model.answer(question, output),
            Self::D3(model) => model.answer(question, output),
        }
    }
}

pub struct Laya {
    tokenizer: Tokenizer,
    config: Config,
    max_input_length: usize,
    cls: u32,
    sep: u32,
    mask: u32,
}

struct Question {
    id: String,
    kind: String,
    labels: Vec<String>,
    criteria: Value,
    compute_chars: usize,
    encoding: ValidEncoding,
    input: DecisionInput,
}

impl Laya {
    pub fn load(
        path: &Path,
        mut tokenizer: Tokenizer,
        max_input_length: usize,
    ) -> anyhow::Result<Self> {
        let config: Config =
            serde_json::from_slice(&std::fs::read(path.join("rl_agent_config.json"))?)?;
        anyhow::ensure!(
            config.max_len > 0 && config.head_max_len > 0,
            "Invalid Laya token budgets"
        );
        for (name, temperature) in config
            .temperature
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("temperature[{i}]"), t))
            .chain(
                config
                    .temperature_by_options
                    .iter()
                    .map(|(key, t)| (key.clone(), t)),
            )
        {
            if !temperature.is_finite() || !(0.5..=5.0).contains(temperature) {
                tracing::warn!(bucket = %name, temperature = %temperature,
                    "Laya temperature will be clamped to [0.5, 5] or replaced with 1; calibration for this bucket is not verified");
            }
        }
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let id = |name| {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| anyhow::anyhow!("Laya tokenizer is missing {name}"))
        };
        let (cls, sep, mask) = (id("[CLS]")?, id("[SEP]")?, id("[MASK]")?);
        Ok(Self {
            tokenizer,
            config,
            max_input_length,
            cls,
            sep,
            mask,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(text.replace("[MASK]", " "), false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }

    fn prepare(&self, request: SystemOneRequest) -> Result<Vec<Question>, String> {
        let state = match request.state {
            ModelInput::Text(text) => text,
            ModelInput::Messages(_) => {
                return Err(
                    "Laya does not support native messages; provide a plain text state".into(),
                );
            }
        };
        // Jev model aliases are accepted; a infinity_v3 deployment serves its configured checkpoint.
        let _model_alias = request.model;
        let max_len = request
            .max_len
            .unwrap_or(self.config.max_len)
            .min(self.max_input_length);
        if request
            .max_len
            .is_some_and(|v| v == 0 || v > self.max_input_length)
        {
            return Err(format!(
                "max_len must be between 1 and {}",
                self.max_input_length
            ));
        }
        let head_max_len = request
            .head_max_len
            .unwrap_or(self.config.head_max_len.min(self.max_input_length));
        if head_max_len == 0 || head_max_len > self.max_input_length {
            return Err(format!(
                "head_max_len must be between 1 and {}",
                self.max_input_length
            ));
        }
        if state.chars().count() > 50_000 {
            return Err("state exceeds 50000 characters".into());
        }
        let state_chars = state.replace("[MASK]", " ").chars().count();
        let state_ids = self.encode(&state)?;
        let mut total_options = 0;
        request
            .questions
            .into_iter()
            .map(|(id, q)| {
                let fields = q.as_object().ok_or("Each question must be an object")?;
                if let Some(field) = fields.keys().find(|key| {
                    !matches!(
                        key.as_str(),
                        "type" | "instructions" | "criteria" | "labels"
                    )
                }) {
                    return Err(format!("Question {id}: unsupported field {field}"));
                }
                let kind = q
                    .get("type")
                    .and_then(Value::as_str)
                    .ok_or("Question type is required")?;
                let question_type = match kind {
                    "choice" => 0,
                    "score" => 1,
                    "noul" => 2,
                    _ => return Err("Question type must be choice, score, or noul".into()),
                };
                let instructions = q
                    .get("instructions")
                    .and_then(Value::as_str)
                    .ok_or("Question instructions must be a string")?;
                if kind != "noul" && q.get("labels").is_some() {
                    return Err("labels is only supported for noul".into());
                }
                let criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
                let mut labels = Vec::new();
                let options: Vec<String> = match kind {
                    "choice" => match &criteria {
                        Value::Object(map) => map
                            .iter()
                            .map(|(k, v)| {
                                labels.push(k.clone());
                                if v.is_null() || v.as_str() == Some("") {
                                    k.clone()
                                } else {
                                    format!("{k}: {}", render(v))
                                }
                            })
                            .collect(),
                        Value::Array(values) => values
                            .iter()
                            .map(|v| {
                                let label = v
                                    .as_str()
                                    .ok_or("Choice labels must be strings")?
                                    .to_owned();
                                labels.push(label.clone());
                                Ok(label)
                            })
                            .collect::<Result<_, String>>()?,
                        _ => return Err("choice criteria must be an object or array".into()),
                    },
                    "score" => criteria
                        .as_array()
                        .ok_or("score criteria must be an array")?
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            labels.push(i.to_string());
                            format!("level {i}: {}", render(v))
                        })
                        .collect(),
                    _ => {
                        if !criteria.is_null()
                            && !criteria
                                .as_object()
                                .is_some_and(|m| m.keys().all(|k| k == "false" || k == "true"))
                        {
                            return Err("noul criteria must contain only false and true".into());
                        }
                        let mut names = ["false".to_owned(), "true".to_owned()];
                        if let Some(custom) = q.get("labels") {
                            let m = custom.as_object().ok_or("noul labels must be an object")?;
                            if m.len() != 2 {
                                return Err("noul labels must contain false and true".into());
                            }
                            for (i, key) in ["false", "true"].iter().enumerate() {
                                names[i] = m
                                    .get(*key)
                                    .and_then(Value::as_str)
                                    .ok_or("noul labels must contain false and true strings")?
                                    .trim()
                                    .into();
                            }
                            if names[0].is_empty() || names[1].is_empty() || names[0] == names[1] {
                                return Err("noul labels must be distinct nonempty strings".into());
                            }
                        }
                        ["false", "true"]
                            .iter()
                            .enumerate()
                            .map(|(i, key)| {
                                labels.push((*key).into());
                                let description = criteria
                                    .get(*key)
                                    .filter(|v| !v.is_null() && v.as_str() != Some(""))
                                    .map(render)
                                    .unwrap_or_else(|| {
                                        if i == 0 {
                                            "no, the statement does not hold".into()
                                        } else {
                                            "yes, the statement holds".into()
                                        }
                                    });
                                format!("{}: {description}", names[i])
                            })
                            .collect()
                    }
                };
                let limit = if kind == "score" { 32 } else { 100 };
                if options.is_empty() || options.len() > limit {
                    return Err(format!("{kind} requires 1 to {limit} options"));
                }
                if labels.iter().any(String::is_empty)
                    || labels.iter().collect::<HashSet<_>>().len() != labels.len()
                {
                    return Err("Option labels must be nonempty and distinct".into());
                }
                total_options += options.len();
                if total_options > 512 {
                    return Err("At most 512 total options are allowed".into());
                }
                let head_text = format!("{kind} question: {instructions}");
                // Like infinity_v3's other endpoints, count input text before truncation,
                // excluding inserted special tokens. Each question repeats the state.
                let compute_chars = state_chars
                    + head_text.replace("[MASK]", " ").chars().count()
                    + options
                        .iter()
                        .map(|o| 1 + o.replace("[MASK]", " ").chars().count())
                        .sum::<usize>();
                let mut head = self.encode(&head_text)?;
                let mut options = options
                    .iter()
                    .map(|o| {
                        let mut ids = self.encode(&format!(" {o}"))?;
                        ids.truncate(48);
                        ids.insert(0, self.mask);
                        Ok(ids)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let mut option_len = options.iter().map(Vec::len).sum::<usize>();
                if head_max_len as isize - (option_len as isize) < 16 {
                    let per = (head_max_len.saturating_sub(16) / options.len()).max(4);
                    for option in &mut options {
                        option.truncate(per);
                    }
                    option_len = options.iter().map(Vec::len).sum();
                }
                head.truncate(head_max_len.saturating_sub(option_len).max(8));
                // Preserve Laya's minimum instruction/option spans, but reject
                // budgets that cannot accommodate them instead of exceeding the override.
                if head.len() + option_len > head_max_len {
                    return Err(format!(
                        "Question {id}: head_max_len cannot fit instructions and all options"
                    ));
                }
                if options.iter().collect::<HashSet<_>>().len() != options.len() {
                    return Err(format!(
                        "Question {id}: token budget makes options indistinguishable"
                    ));
                }
                let mut ids = vec![self.cls];
                ids.extend(head);
                ids.push(self.sep);
                let mut markers = Vec::new();
                for option in options {
                    markers.push(ids.len());
                    ids.extend(option);
                }
                ids.push(self.sep);
                if ids.len() + 1 > max_len {
                    return Err(format!("Question {id}: max_len cannot fit all options"));
                }
                let room = (max_len - ids.len() - 1).min(state_ids.len());
                ids.extend_from_slice(&state_ids[..room]);
                ids.push(self.sep);
                let length = ids.len();
                Ok(Question {
                    id,
                    kind: kind.into(),
                    labels,
                    criteria,
                    compute_chars,
                    encoding: ValidEncoding {
                        multimodal: None,
                        input_ids: ids,
                        token_type_ids: vec![0; length],
                        position_ids: (0..length as u32).collect(),
                        tokens: vec![],
                        offsets: vec![],
                    },
                    input: DecisionInput::Laya {
                        question_type,
                        markers,
                    },
                })
            })
            .collect()
    }

    fn answer(
        &self,
        question: &Question,
        output: DecisionOutput,
    ) -> Result<ResultValue, AnswerError> {
        let DecisionInput::Laya { question_type, .. } = &question.input else {
            return Err("Invalid Laya metadata".into());
        };
        let k = question.labels.len();
        if output.logits.len() != k
            || !output.logits.iter().all(|v| v.is_finite())
            || !output.action_probability.is_finite()
        {
            return Err("Model returned invalid decision scores".into());
        }
        let bucket = match k {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        let t = self
            .config
            .temperature_by_options
            .get(&format!("{}:{bucket}", question.kind))
            .or_else(|| self.config.temperature.get(*question_type))
            .copied()
            .unwrap_or(1.0);
        let t = if t.is_finite() {
            t.clamp(0.5, 5.0)
        } else {
            1.0
        };
        let max = output
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut p = super::result_storage::vector(output.logits.len())?;
        p.extend(output.logits.iter().map(|v| (((v - max) / t) as f64).exp()));
        let sum: f64 = p.iter().sum();
        for value in &mut p {
            *value /= sum;
        }
        let best = (0..k)
            .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        let confidence = if k == 1 {
            1.0
        } else {
            1.0 + p.iter().map(|v| v * v.max(1e-9).ln()).sum::<f64>() / (k as f64).ln()
        };
        let mut answer = ResultValue::object(8)?;
        answer.put("type", ResultValue::string(&question.kind)?)?;
        answer.put("confidence", ResultValue::number(round(confidence)))?;
        answer.put("answer_confidence", ResultValue::number(round(p[best])))?;
        answer.put(
            "action",
            ResultValue::fields([(
                "act_probability",
                ResultValue::number(round(output.action_probability as f64)),
            )])?,
        )?;
        match question.kind.as_str() {
            "choice" => answer.put("choice", ResultValue::string(&question.labels[best])?)?,
            "score" => {
                answer.put(
                    "score",
                    ResultValue::number(round(
                        p.iter().enumerate().map(|(i, p)| i as f64 * p).sum(),
                    )),
                )?;
                answer.put(
                    "legend",
                    ResultValue::legend(question.criteria.as_array().unwrap())?,
                )?;
            }
            _ => {
                answer.put("noul", ResultValue::number(round(p[1])))?;
                answer.put("confidence", ResultValue::number(round(p[best])))?;
            }
        }
        if question.kind != "noul" {
            for value in &mut p {
                *value = round(*value);
            }
            answer.put(
                "probabilities",
                ResultValue::probabilities(question.labels.iter().map(String::as_str), &p)?,
            )?;
        }
        Ok(answer)
    }
}

fn round(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

// Match Python json.dumps' spacing without changing text inside JSON strings.
fn render(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(v) => format!(
            "[{}]",
            v.iter().map(json_text).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), json_text(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => value.to_string(),
    }
}
fn json_text(value: &Value) -> String {
    if value.is_string() {
        value.to_string()
    } else {
        render(value)
    }
}

type ApiError = (StatusCode, Json<crate::ErrorResponse>);
fn error(status: StatusCode, message: impl ToString) -> ApiError {
    metrics::counter!("infinity_request_failure", "err" => "systemone").increment(1);
    let error_type = match status {
        StatusCode::TOO_MANY_REQUESTS => crate::ErrorType::Overloaded,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => crate::ErrorType::Validation,
        _ => crate::ErrorType::Backend,
    };
    (
        status,
        Json(crate::ErrorResponse {
            error: message.to_string(),
            error_type,
        }),
    )
}

#[utoipa::path(post, path = "/v1/systemone", request_body = Value,
    responses((status = 200, description = "Typed decisions", body = Value),
              (status = 422, description = "Invalid question or token budget"),
              (status = 429, description = "Server overloaded")))]
pub async fn systemone(
    super::model_routing::RoutedJson { runtime, value }: super::model_routing::RoutedJson<Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let result = systemone_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        Extension(runtime.systemone.clone()),
        Json(value),
    )
    .await;
    match result {
        Ok((headers, Json(mut value))) => {
            if runtime.named {
                if let Err(failure) = ResultValue::string(&runtime.info.model_id)
                    .and_then(|model| value.put("model", model))
                {
                    return error(StatusCode::TOO_MANY_REQUESTS, failure).into_response();
                }
            }
            (headers, Json(value)).into_response()
        }
        Err(error) => error.into_response(),
    }
}

pub(crate) async fn systemone_inner(
    Extension(infer): Extension<Infer>,
    Extension(info): Extension<crate::Info>,
    Extension(service): Extension<Option<Arc<SystemOne>>>,
    Json(raw_request): Json<Value>,
) -> Result<(HeaderMap, Json<ResultValue>), ApiError> {
    metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "systemone").increment(1);
    let service = service.ok_or_else(|| {
        error(
            StatusCode::BAD_REQUEST,
            "Loaded model does not support typed decisions",
        )
    })?;
    let (mut request, d3_images, d3_videos) = if matches!(service.as_ref(), SystemOne::D3(_)) {
        serde_json::from_value::<d3::Request>(raw_request)
            .map_err(|e| error(StatusCode::UNPROCESSABLE_ENTITY, e))?
            .into_common()
            .map_err(|e| error(StatusCode::UNPROCESSABLE_ENTITY, e))?
    } else {
        (
            serde_json::from_value::<SystemOneRequest>(raw_request)
                .map_err(|e| error(StatusCode::UNPROCESSABLE_ENTITY, e))?,
            vec![],
            vec![],
        )
    };
    let d3_max_len = request.max_len;
    if request.questions.len() > info.max_decision_questions {
        return Err(error(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "At most {} questions are allowed by max-decision-questions",
                info.max_decision_questions
            ),
        ));
    }
    // Hold one infinity_v3 admission permit for the entire HTTP request, including tokenization.
    let _permit = infer
        .try_acquire_permit()
        .map_err(|e| error(StatusCode::TOO_MANY_REQUESTS, e))?;
    metrics::counter!("infinity_systemone_count", "model" => info.model_id.clone()).increment(1);
    let start = Instant::now();
    // Validate and tokenize questions before fetching media, off the async executor.
    let worker = service.clone();
    let (prepared, _input_memory) = infinity_v3_core::input_memory::spawn_blocking(move || {
        let sources = match worker.as_ref() {
            SystemOne::Rune(_) => rune::Rune::image_sources(&mut request)?,
            SystemOne::Onejev(_) => onejev::Onejev::image_sources(&mut request)?,
            _ => vec![],
        };
        let max_len = request.max_len;
        Ok::<_, String>((worker.prepare(request)?, sources, max_len))
    })
    .await
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let (mut questions, sources, max_len) =
        prepared.map_err(|e| error(StatusCode::UNPROCESSABLE_ENTITY, e))?;
    if !sources.is_empty() {
        let media_error = |e: infinity_v3_core::TextEmbeddingsError| {
            let status = if matches!(e, infinity_v3_core::TextEmbeddingsError::Overloaded(_)) {
                StatusCode::TOO_MANY_REQUESTS
            } else {
                StatusCode::UNPROCESSABLE_ENTITY
            };
            error(status, e)
        };
        let attached = match service.as_ref() {
            SystemOne::Rune(model) => {
                let images = model.prepare_images(sources).await.map_err(media_error)?;
                model.attach_images(&mut questions, images, max_len)
            }
            SystemOne::Onejev(model) => {
                let images = model.prepare_images(sources).await.map_err(media_error)?;
                model.attach_images(&mut questions, images, max_len)
            }
            _ => unreachable!("image sources require a multimodal decision model"),
        };
        attached.map_err(|e| error(StatusCode::UNPROCESSABLE_ENTITY, e))?;
    }
    if !d3_images.is_empty() || !d3_videos.is_empty() {
        let SystemOne::D3(model) = service.as_ref() else {
            unreachable!()
        };
        let media = model
            .prepare_media(d3_images, d3_videos)
            .await
            .map_err(|e| {
                let status = if matches!(e, infinity_v3_core::TextEmbeddingsError::Overloaded(_)) {
                    StatusCode::TOO_MANY_REQUESTS
                } else {
                    StatusCode::UNPROCESSABLE_ENTITY
                };
                error(status, e)
            })?;
        model
            .attach_media(&mut questions, media, d3_max_len)
            .map_err(|e| error(StatusCode::UNPROCESSABLE_ENTITY, e))?;
    }
    let tokenization = start.elapsed();
    let compute_chars = questions.iter().map(|q| q.compute_chars).sum();
    let input_tokens: usize = questions.iter().map(|q| q.encoding.input_ids.len()).sum();
    let batch_counter = Arc::new(std::sync::atomic::AtomicUsize::new(questions.len()));
    let output_tokens = if matches!(
        service.as_ref(),
        SystemOne::Rune(_) | SystemOne::Onejev(_) | SystemOne::Pplx(_)
    ) {
        questions.len()
    } else {
        0
    };
    let response_model = if !matches!(service.as_ref(), SystemOne::Laya(_)) {
        info.model_id.as_str()
    } else {
        "laya-rl-agent"
    };
    let answers = futures::future::try_join_all(questions.into_iter().map(|mut question| {
        let infer = infer.clone();
        let service = service.clone();
        let batch_counter = batch_counter.clone();
        async move {
            let encoding = std::mem::replace(
                &mut question.encoding,
                ValidEncoding {
                    multimodal: None,
                    input_ids: vec![],
                    token_type_ids: vec![],
                    position_ids: vec![],
                    tokens: vec![],
                    offsets: vec![],
                },
            );
            let output = infer
                .decide(
                    encoding,
                    question.input.clone(),
                    tokenization,
                    batch_counter,
                )
                .await
                .map_err(|e| {
                    let status = if matches!(
                        e,
                        infinity_v3_core::TextEmbeddingsError::Backend(
                            infinity_v3_backend::BackendError::Overloaded(_)
                        )
                    ) {
                        StatusCode::TOO_MANY_REQUESTS
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    };
                    error(status, e)
                })?;
            let answer = service.answer(&question, output.results).map_err(|e| {
                let status = if matches!(e, AnswerError::Memory(_)) {
                    StatusCode::TOO_MANY_REQUESTS
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                error(status, e)
            })?;
            Ok::<_, ApiError>((question.id, answer, output.metadata))
        }
    }))
    .await?;
    let count = answers.len().max(1) as u32;
    let queue = answers.iter().map(|(_, _, m)| m.queue).sum::<Duration>() / count;
    let inference = answers
        .iter()
        .map(|(_, _, m)| m.inference)
        .sum::<Duration>()
        / count;
    let metadata = crate::ResponseMetadata::new(
        compute_chars,
        input_tokens,
        start,
        tokenization,
        queue,
        inference,
    );
    metadata.record_metrics(&info.model_id);
    metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "systemone").increment(1);
    let answers = if matches!(service.as_ref(), SystemOne::Clef(_)) {
        match answers.into_iter().next().unwrap().1 {
            ResultValue::Object(answers) => ResultValue::Object(answers),
            _ => {
                return Err(error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Invalid joint answers",
                ));
            }
        }
    } else {
        let mut object = ResultValue::object(answers.len())
            .map_err(|e| error(StatusCode::TOO_MANY_REQUESTS, e))?;
        for (id, answer, _) in answers {
            object
                .put(&id, answer)
                .map_err(|e| error(StatusCode::TOO_MANY_REQUESTS, e))?;
        }
        object
    };
    let response = ResultValue::fields([
        (
            "model",
            ResultValue::string(response_model)
                .map_err(|e| error(StatusCode::TOO_MANY_REQUESTS, e))?,
        ),
        ("answers", answers),
        (
            "usage",
            ResultValue::fields([
                ("input_tokens", ResultValue::integer(input_tokens)),
                ("output_tokens", ResultValue::integer(output_tokens)),
            ])
            .map_err(|e| error(StatusCode::TOO_MANY_REQUESTS, e))?,
        ),
    ])
    .map_err(|e| error(StatusCode::TOO_MANY_REQUESTS, e))?;
    Ok((metadata.into(), Json(response)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service() -> Laya {
        use tokenizers::models::wordlevel::WordLevel;
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab([("[UNK]".to_owned(), 0)].into_iter().collect())
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        );
        Laya {
            tokenizer,
            config: Config {
                max_len: 1024,
                head_max_len: 256,
                temperature: vec![1.; 3],
                temperature_by_options: HashMap::new(),
            },
            max_input_length: 1024,
            cls: 1,
            sep: 2,
            mask: 3,
        }
    }

    #[test]
    fn json_render_preserves_strings_spacing_and_order() {
        let value: Value = serde_json::from_str(r#"{"z":"a,b:c","a":["é",{"x":false}]}"#).unwrap();
        assert_eq!(
            render(&value),
            r#"{"z": "a,b:c", "a": ["é", {"x": false}]}"#
        );
    }

    #[test]
    fn unsupported_execution_semantics_are_not_silently_ignored() {
        for (field, value) in [
            ("think", json!(64)),
            ("messages", json!([])),
            ("mode", json!("joint")),
        ] {
            let mut request = json!({"state":"hello", "questions":{}});
            request[field] = value;
            assert!(serde_json::from_value::<SystemOneRequest>(request).is_err());
        }
        for (field, value) in [
            ("depends_on", json!(["first"])),
            ("ask_if", json!({"first":[true]})),
            ("alone", json!(true)),
        ] {
            let mut request = json!({"state":"hello", "questions":{"second":{"type":"noul","instructions":"Is it true?"}}});
            request["questions"]["second"][field] = value;
            let err = service()
                .prepare(serde_json::from_value(request).unwrap())
                .err()
                .unwrap();
            assert!(err.contains(field), "{err}");
        }
    }

    #[test]
    fn laya_rejects_native_messages_before_inference() {
        let service = service();
        for content in [
            json!("hello"),
            json!([{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]),
        ] {
            let request =
                json!({"state":{"messages":[{"role":"user","content":content}]}, "questions":{}});
            let error = service
                .prepare(serde_json::from_value(request).unwrap())
                .err()
                .unwrap();
            assert!(error.contains("does not support native messages"));
        }
    }

    #[test]
    fn calibrated_answers_use_option_order_and_typed_semantics() {
        let mut service = service();
        let fixture: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/laya-systemone.json")).unwrap();
        service.config = serde_json::from_value(json!({"max_len":1024,"head_max_len":256,"temperature_by_options":{"choice:2":1.9063563346862793,"score:3-5":1.2514300346374512,"noul:2":1.983399510383606}})).unwrap();
        for seq in fixture["sequences"].as_array().unwrap() {
            let id = seq["id"].as_str().unwrap();
            let q = &fixture["request"]["questions"][id];
            let kind = q["type"].as_str().unwrap();
            let labels = match kind {
                "choice" => q["criteria"].as_object().unwrap().keys().cloned().collect(),
                "score" => vec!["0".into(), "1".into(), "2".into()],
                _ => vec!["false".into(), "true".into()],
            };
            let question = Question {
                id: id.into(),
                kind: kind.into(),
                labels,
                criteria: q["criteria"].clone(),
                compute_chars: 0,
                input: DecisionInput::Laya {
                    question_type: match kind {
                        "choice" => 0,
                        "score" => 1,
                        _ => 2,
                    },
                    markers: vec![],
                },
                encoding: ValidEncoding {
                    multimodal: None,
                    input_ids: vec![],
                    token_type_ids: vec![],
                    position_ids: vec![],
                    tokens: vec![],
                    offsets: vec![],
                },
            };
            let output = DecisionOutput {
                memory: None,
                logits: serde_json::from_value(seq["logits"].clone()).unwrap(),
                action_probability: seq["action_probability"].as_f64().unwrap() as f32,
            };
            assert_eq!(
                service.answer(&question, output).unwrap(),
                fixture["response"]["answers"][id]
            );
        }
    }

    #[test]
    fn compute_characters_include_each_question_and_repeated_state() {
        let service = service();
        let request = json!({"state":"é[MASK]", "questions": {
            "team": {"type":"choice", "instructions":"Pick A", "criteria":{"billing":"payments"}},
            "level": {"type":"score", "instructions":"Rate", "criteria":["low"]}
        }});
        let questions = service
            .prepare(serde_json::from_value(request).unwrap())
            .ok()
            .unwrap();
        assert_eq!(
            questions[0].compute_chars,
            "é choice question: Pick A billing: payments"
                .chars()
                .count()
        );
        assert_eq!(
            questions[1].compute_chars,
            "é score question: Rate level 0: low".chars().count()
        );
        let request = json!({"state":"", "questions":{
            "team":{"type":"choice", "instructions":"Pick", "criteria":["billing"]}
        }});
        let questions = service
            .prepare(serde_json::from_value(request).unwrap())
            .ok()
            .unwrap();
        assert_eq!(
            questions[0].compute_chars,
            "choice question: Pick billing".chars().count()
        );
    }

    #[test]
    fn rejects_head_budget_smaller_than_minimum_option_spans() {
        let mut service = service();
        service.tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        let request = json!({
            "state": "text", "max_len": 1024, "head_max_len": 16,
            "questions": {"x": {
                "type": "choice", "instructions": "Choose the most appropriate category for this request",
                "criteria": ["alpha option description", "beta option description", "gamma option description", "delta option description", "epsilon option description"]
            }}
        });
        let error = service
            .prepare(serde_json::from_value(request).unwrap())
            .err()
            .unwrap();
        assert!(error.contains("head_max_len cannot fit"), "{error}");
    }

    #[test]
    fn rejects_invalid_state_types_labels_and_budgets() {
        let service = service();
        for request in [
            json!({"state":"text","questions":{},"max_len":0}),
            json!({"state":"text","questions":{},"head_max_len":2048}),
            json!({"state":"text","questions":{"x":{"type":"choice","instructions":"choose","criteria":[]}}}),
            json!({"state":"text","questions":{"x":{"type":"noul","instructions":"yes?","labels":{"false":"x","true":"x"}}}}),
            json!({"state":"text","questions":{"x":{"type":"score","instructions":"score","criteria":"invalid"}}}),
        ] {
            assert!(service
                .prepare(serde_json::from_value(request).unwrap())
                .is_err());
        }
    }

    #[test]
    fn checkpoint_tokenization_matches_upstream_fixture() -> anyhow::Result<()> {
        let Ok(path) = std::env::var("LAYA_CHECKPOINT_DIR") else {
            return Ok(());
        };
        let path = Path::new(&path);
        let tokenizer = Tokenizer::from_file(path.join("tokenizer/tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let service = Laya::load(path, tokenizer, 1024)?;
        let fixture: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/laya-systemone.json"))?;
        let questions = service
            .prepare(serde_json::from_value(fixture["request"].clone())?)
            .map_err(anyhow::Error::msg)?;
        for (question, expected) in questions
            .iter()
            .zip(fixture["sequences"].as_array().unwrap())
        {
            assert_eq!(json!(question.encoding.input_ids), expected["ids"]);
            let DecisionInput::Laya { markers, .. } = &question.input else {
                panic!("expected Laya metadata")
            };
            assert_eq!(json!(markers), expected["markers"]);
        }
        let mut request = fixture["request"].clone();
        request["max_len"] = json!(8);
        assert!(service.prepare(serde_json::from_value(request)?).is_err());
        Ok(())
    }
}

impl super::model_routing::RequestedModel for SystemOneRequest {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        Ok(self.model.as_deref())
    }
}
