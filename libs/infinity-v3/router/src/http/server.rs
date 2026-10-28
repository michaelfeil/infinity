// Fork license metadata updated by Infinity Core Contributors; see LICENSE.
use super::model_routing::{RoutedJson, RoutingRejection};
use super::response_format::Json;
use super::tensor::{encode_matrix, encode_vector, Tensor};
use crate::http::ner::apply_aggregation;
/// HTTP Server logic
use crate::http::types::{
    ClassifyData, ClassifyErrorResponse, ClassifyRequest, ClassifyRequestId, ClassifyResponse,
    ClassifyUsage, DecodeRequest, DecodeResponse, EmbedAllRequest, EmbedAllResponse, EmbedRequest,
    EmbedResponse, EmbedSparseRequest, EmbedSparseResponse, Embedding, EmbeddingInput,
    EncodingFormat, InputBatch, InputIds, OpenAICompatEmbedding, OpenAICompatErrorResponse,
    OpenAICompatRequest, OpenAICompatResponse, OpenAICompatUsage, PredictInput, PredictRequest,
    PredictResponse, PredictTokensRequest, Prediction, Rank, RerankRequest, RerankResponse,
    Sequence, ServiceTier, SimilarityInput, SimilarityParameters, SimilarityRequest,
    SimilarityResponse, SimpleToken, SparseValue, TokenPredictResponse, TokenizeInput,
    TokenizeRequest, TokenizeResponse, TruncationDirection,
};
use crate::model_registry::ModelRegistry;
use crate::{
    logging, shutdown, ClassifierModel, EmbeddingModel, ErrorResponse, ErrorType, Info, ModelType,
    ResponseMetadata,
};
use ::http::HeaderMap;
use anyhow::Context;
use axum::extract::{DefaultBodyLimit, Extension};
use axum::http::HeaderValue;
use axum::http::{Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{http, Router};
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;
use futures::future::join_all;
use http::header::AUTHORIZATION;
use infinity_v3_backend::BackendError;
use infinity_v3_core::infer::{
    AllEmbeddingsInferResponse, Infer, InferMetadata, PooledEmbeddingsInferResponse,
};
use infinity_v3_core::tokenization::{into_tokens, SimpleToken as CoreSimpleToken};
use infinity_v3_core::TextEmbeddingsError;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use simsimd::SpatialSimilarity;
use std::net::SocketAddr;
use std::sync::{atomic::AtomicUsize, Arc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::OwnedSemaphorePermit;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

///infinity_v3 endpoint info
#[utoipa::path(
get,
tag = "infinity_v3",
path = "/info",
params(("model" = Option<String>, Query, description = "Served model name; required when multiple models are deployed")),
responses((status = 200, description = "Served model info", body = Info), (status = 400, description = "Missing or unknown model", body = ErrorResponse))
)]
#[instrument(skip_all)]
async fn get_model_info(
    Extension(registry): Extension<ModelRegistry>,
    axum::extract::Query(query): axum::extract::Query<ModelQuery>,
) -> Response {
    match registry.resolve(query.model.as_deref()) {
        Ok(model) => Json(model.info.clone()).into_response(),
        Err(message) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: message,
                error_type: ErrorType::Empty,
            }),
        )
            .into_response(),
    }
}

#[derive(serde::Deserialize)]
struct ModelQuery {
    model: Option<String>,
}

#[derive(serde::Serialize, utoipa::ToSchema)]
struct ServedModel {
    id: String,
    object: &'static str,
    checkpoint_id: String,
    backend: &'static str,
    capabilities: Vec<&'static str>,
}
#[derive(serde::Serialize, utoipa::ToSchema)]
struct ServedModels {
    object: &'static str,
    data: Vec<ServedModel>,
}

#[utoipa::path(get, tag="infinity_v3", path="/v1/models", responses((status=200,description="Resident models and capabilities",body=ServedModels)))]
async fn models(Extension(registry): Extension<ModelRegistry>) -> Json<ServedModels> {
    let data = registry
        .models()
        .iter()
        .map(|model| ServedModel {
            id: model.info.model_id.clone(),
            object: "model",
            checkpoint_id: model.checkpoint_id.clone(),
            backend: match model.backend_kind {
                infinity_v3_backend::BackendKind::Candle => "candle",
                infinity_v3_backend::BackendKind::Libtorch => "libtorch",
            },
            capabilities: match &model.info.model_type {
                ModelType::Embedding(model) if model.pooling == "splade" => {
                    vec!["embed_sparse", "tokenize", "decode"]
                }
                ModelType::Embedding(_) => {
                    vec!["embed", "embed_all", "similarity", "tokenize", "decode"]
                }
                ModelType::Classifier(_) => vec!["classify", "tokenize", "decode"],
                ModelType::Reranker(_) => vec!["rerank", "classify", "tokenize", "decode"],
                ModelType::Decision => vec!["systemone", "tokenize", "decode"],
            },
        })
        .collect();
    Json(ServedModels {
        object: "list",
        data,
    })
}

#[utoipa::path(
get,
tag = "infinity_v3",
path = "/health",
responses(
(status = 200, description = "Everything is working fine"),
(status = 500, description = "One or more deployed models are unhealthy", body = ErrorResponse,
example = json ! ({"error": "unhealthy", "error_type": "unhealthy"})),
)
)]
#[instrument(skip_all)]
/// Health check method
async fn health(
    Extension(registry): Extension<ModelRegistry>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    match registry.healthy().await {
        true => Ok(()),
        false => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "One or more deployed models are unhealthy".to_string(),
                error_type: ErrorType::Unhealthy,
            }),
        )),
    }
}

/// Deprecated: use /v1/classify. This route retains the legacy label/score response.
#[utoipa::path(
post,
tag = "infinity_v3",
path = "/predict",
request_body = PredictRequest,
responses(
(status = 200, description = "Predictions", body = PredictResponse),
(status = 424, description = "Prediction Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn predict(
    context: Extension<Option<opentelemetry::Context>>,
    RoutedJson { runtime, value }: RoutedJson<PredictRequest>,
) -> Response {
    predict_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        context,
        Json(value),
    )
    .await
    .into_response()
}

async fn predict_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Extension(context): Extension<Option<opentelemetry::Context>>,
    Json(req): Json<PredictRequest>,
) -> Result<(HeaderMap, Json<PredictResponse>), (StatusCode, Json<ErrorResponse>)> {
    let single = matches!(&req.inputs, PredictInput::Single(_));
    let labels = classifier_labels(&info.model_type)?;
    let (scores, metadata) = run_classification(infer, info, Extension(context), req).await?;
    reserve_classification_output(&scores, &labels).map_err(ErrorResponse::from)?;
    let predictions = scores
        .into_iter()
        .map(|scores| {
            let mut predictions: Vec<_> = scores
                .into_iter()
                .enumerate()
                .map(|(index, score)| Prediction {
                    score,
                    label: labels
                        .get(&index.to_string())
                        .cloned()
                        .unwrap_or_else(|| format!("LABEL_{index}")),
                })
                .collect();
            predictions.sort_by(|a, b| a.score.total_cmp(&b.score));
            predictions.reverse();
            predictions
        })
        .collect::<Vec<_>>();
    let response = if single {
        PredictResponse::Single(predictions.into_iter().next().unwrap())
    } else {
        PredictResponse::Batch(predictions)
    };
    Ok((HeaderMap::from(metadata), Json(response)))
}

fn classifier_labels(
    model_type: &ModelType,
) -> Result<std::collections::HashMap<String, String>, ErrorResponse> {
    match model_type {
        ModelType::Classifier(classifier) | ModelType::Reranker(classifier) => {
            Ok(classifier.id2label.clone())
        }
        _ => Err(ErrorResponse {
            error: "Model is not a classifier model".into(),
            error_type: ErrorType::Backend,
        }),
    }
}

async fn run_classification(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Extension(context): Extension<Option<opentelemetry::Context>>,
    req: PredictRequest,
) -> Result<(Vec<Vec<f32>>, ResponseMetadata), ErrorResponse> {
    let infer = Extension(infer.0.with_service_tier(req.service_tier));
    let service_tier = infer.service_tier();
    let span = tracing::Span::current();
    if let Some(context) = context {
        span.set_parent(context);
    }

    let start_time = Instant::now();

    // Closure for predict
    let predict_inner = move |inputs: Sequence,
                              truncate: bool,
                              infer: Infer,
                              permit: Option<OwnedSemaphorePermit>,
                              _batch_counter: Option<Arc<AtomicUsize>>| async move {
        let permit = match permit {
            None => infer.acquire_permit().await,
            Some(permit) => permit,
        };

        let response = infer
            .predict(
                inputs,
                truncate,
                req.truncation_direction.into(),
                req.raw_scores,
                permit,
                None,
            )
            .await
            .map_err(ErrorResponse::from)?;

        if response.results.is_empty() || response.results.iter().any(|score| !score.is_finite()) {
            return Err(ErrorResponse {
                error: "Classifier returned empty or non-finite scores".into(),
                error_type: ErrorType::Backend,
            });
        }

        Ok::<(usize, Duration, Duration, Duration, Vec<f32>), ErrorResponse>((
            response.metadata.prompt_tokens,
            response.metadata.tokenization,
            response.metadata.queue,
            response.metadata.inference,
            response.results,
        ))
    };

    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = match req.inputs {
        PredictInput::Single(inputs) => {
            let counter = metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "single");
            counter.increment(1);

            let compute_chars = inputs.count_chars();
            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let (prompt_tokens, tokenization, queue, inference, predictions) =
                predict_inner(inputs, truncate, infer.0, Some(permit), None).await?;

            let counter = metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "single");
            counter.increment(1);

            (
                vec![predictions],
                ResponseMetadata::new(
                    compute_chars,
                    prompt_tokens,
                    start_time,
                    tokenization,
                    queue,
                    inference,
                    service_tier,
                ),
            )
        }
        PredictInput::Batch(inputs) => {
            let counter = metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "batch");
            counter.increment(1);

            let batch_size = inputs.len();
            if batch_size == 0 {
                return Err(ErrorResponse {
                    error: "Input cannot be empty".into(),
                    error_type: ErrorType::Empty,
                });
            }
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let batch_counter = if batch_size == 1 {
                None
            } else {
                Some(Arc::new(AtomicUsize::new(batch_size)))
            };
            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.count_chars();
                let local_infer = infer.clone();
                let local_batch_counter = batch_counter.clone();
                futures.push(predict_inner(
                    input,
                    truncate,
                    local_infer.0,
                    None,
                    local_batch_counter,
                ))
            }
            let results = join_all(futures).await.into_iter().collect::<Result<
                Vec<(usize, Duration, Duration, Duration, Vec<f32>)>,
                ErrorResponse,
            >>()?;

            let mut predictions = Vec::with_capacity(batch_size);
            let mut total_tokenization_time = 0;
            let mut total_queue_time = 0;
            let mut total_inference_time = 0;
            let mut total_compute_tokens = 0;

            for r in results {
                total_compute_tokens += r.0;
                total_tokenization_time += r.1.as_nanos() as u64;
                total_queue_time += r.2.as_nanos() as u64;
                total_inference_time += r.3.as_nanos() as u64;
                predictions.push(r.4);
            }
            let batch_size = batch_size as u64;

            let counter = metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "batch");
            counter.increment(1);

            (
                predictions,
                ResponseMetadata::new(
                    compute_chars,
                    total_compute_tokens,
                    start_time,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                    service_tier,
                ),
            )
        }
    };

    metadata.record_span(&span);
    metadata.record_metrics(&info.model_id);

    tracing::info!("Success");

    Ok((response, metadata))
}

#[utoipa::path(
    post, tag = "infinity_v3", path = "/v1/classify",
    request_body = ClassifyRequest,
    responses(
        (status = 200, description = "Classification probabilities and token usage", body = ClassifyResponse),
        (status = 400, description = "Invalid classification input", body = ClassifyErrorResponse),
        (status = 422, description = "Invalid JSON or tokenization", body = ClassifyErrorResponse),
        (status = 424, description = "Classification backend failure", body = ClassifyErrorResponse),
        (status = 429, description = "Model is overloaded", body = ClassifyErrorResponse),
        (status = 413, description = "Batch or token limit exceeded", body = ClassifyErrorResponse),
    )
)]
#[instrument(skip_all)]
async fn classify(
    context: Extension<Option<opentelemetry::Context>>,
    request: Result<RoutedJson<ClassifyRequest>, RoutingRejection>,
) -> Response {
    let request = match request {
        Ok(request) => request,
        Err(error) => {
            return classify_error(error.status(), error.body_text(), "invalid_request_error")
                .into_response()
        }
    };
    let RoutedJson { runtime, value } = request;
    classify_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        context,
        Ok(Json(value)),
    )
    .await
    .into_response()
}

async fn classify_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    context: Extension<Option<opentelemetry::Context>>,
    request: Result<Json<ClassifyRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<(HeaderMap, Json<ClassifyResponse>), (StatusCode, Json<ClassifyErrorResponse>)> {
    let Json(request) = request.map_err(|error| {
        classify_error(error.status(), error.body_text(), "invalid_request_error")
    })?;
    let labels = classifier_labels(&info.model_type).map_err(classify_inference_error)?;
    let model = info.model_id.clone();
    let activation = request.use_activation != Some(false);
    let request = request.into_predict().map_err(|message| {
        classify_error(
            StatusCode::BAD_REQUEST,
            message.into(),
            "invalid_request_error",
        )
    })?;
    let (scores, metadata) = run_classification(infer, info, context, request)
        .await
        .map_err(classify_inference_error)?;
    reserve_classification_output(&scores, &labels)
        .map_err(ErrorResponse::from)
        .map_err(classify_inference_error)?;
    let response =
        classification_response(scores, &labels, model, metadata.compute_tokens, activation);
    Ok((HeaderMap::from(metadata), Json(response)))
}

fn classification_response(
    scores: Vec<Vec<f32>>,
    labels: &std::collections::HashMap<String, String>,
    model: String,
    prompt_tokens: usize,
    activation: bool,
) -> ClassifyResponse {
    ClassifyResponse {
        id: format!("classify-{}", uuid::Uuid::new_v4().simple()),
        object: "list",
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model,
        data: classification_data(scores, labels, activation),
        usage: ClassifyUsage {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
            prompt_tokens_details: None,
        },
    }
}

fn classification_data(
    scores: Vec<Vec<f32>>,
    labels: &std::collections::HashMap<String, String>,
    activation: bool,
) -> Vec<ClassifyData> {
    scores
        .into_iter()
        .enumerate()
        .map(|(index, mut logits)| {
            // Scores have been validated by the shared inference path. Keep class-ID order
            // and select the first maximum, as SGLang's torch.argmax does.
            let mut best = 0;
            for class in 1..logits.len() {
                if logits[class] > logits[best] {
                    best = class;
                }
            }
            if activation {
                let max = logits[best];
                let sum: f32 = logits
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - max).exp();
                        *score
                    })
                    .sum();
                for score in &mut logits {
                    *score /= sum;
                }
            }
            ClassifyData {
                index,
                label: labels
                    .get(&best.to_string())
                    .cloned()
                    .unwrap_or_else(|| format!("LABEL_{best}")),
                num_classes: logits.len(),
                probs: logits,
            }
        })
        .collect()
}

fn classify_error(
    status: StatusCode,
    message: String,
    error_type: &str,
) -> (StatusCode, Json<ClassifyErrorResponse>) {
    (
        status,
        Json(ClassifyErrorResponse {
            object: "error",
            message,
            error_type: error_type.into(),
            param: None,
            code: status.as_u16(),
        }),
    )
}

pub(crate) fn classify_inference_error(
    error: ErrorResponse,
) -> (StatusCode, Json<ClassifyErrorResponse>) {
    let (status, Json(error)): (StatusCode, Json<ErrorResponse>) = error.into();
    let error_type = match status {
        StatusCode::TOO_MANY_REQUESTS => "overloaded_error",
        StatusCode::BAD_REQUEST
        | StatusCode::UNPROCESSABLE_ENTITY
        | StatusCode::PAYLOAD_TOO_LARGE => "invalid_request_error",
        _ => "server_error",
    };
    classify_error(status, error.error, error_type)
}

async fn deprecate_predict(mut response: axum::response::Response) -> axum::response::Response {
    response
        .headers_mut()
        .insert("deprecation", HeaderValue::from_static("@1791590400"));
    response.headers_mut().append(
        "link",
        HeaderValue::from_static("</v1/classify>; rel=\"successor-version\""),
    );
    response
}

/// Get Token Predictions. Returns a 424 status code if the model is not a Sequence Classification model
#[utoipa::path(
    post,
    tag = "infinity_v3",
    path = "/predict_tokens",
    request_body = PredictTokensRequest,
    responses(
        (status = 200, description = "Token Predictions", body = TokenPredictResponse),
        (status = 424, description = "Prediction Error", body = ErrorResponse,
        example = json ! ({"error": "Inference failed", "error_type": "backend"})),
        (status = 429, description = "Model is overloaded", body = ErrorResponse,
        example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
        (status = 422, description = "Tokenization error", body = ErrorResponse,
        example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
        (status = 400, description = "Batch is empty", body = ErrorResponse,
        example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
        (status = 413, description = "Batch size error", body = ErrorResponse,
        example = json ! ({"error": "Batch size error", "error_type": "validation"})),
    )
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn predict_tokens(
    context: Extension<Option<opentelemetry::Context>>,
    RoutedJson { runtime, value }: RoutedJson<PredictTokensRequest>,
) -> Response {
    predict_tokens_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        context,
        Json(value),
    )
    .await
    .into_response()
}

async fn predict_tokens_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Extension(context): Extension<Option<opentelemetry::Context>>,
    Json(req): Json<PredictTokensRequest>,
) -> Result<(HeaderMap, Json<TokenPredictResponse>), (StatusCode, Json<ErrorResponse>)> {
    let infer = Extension(infer.0.with_service_tier(req.service_tier));
    let service_tier = infer.service_tier();
    let span = tracing::Span::current();
    if let Some(context) = context {
        span.set_parent(context);
    }

    let start_time = Instant::now();

    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = {
        let counter = metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "batch");
        counter.increment(1);

        let batch_size = req.inputs.len();
        if batch_size > info.max_client_batch_size {
            let message = format!(
                "batch size {batch_size} > maximum allowed batch size {}",
                info.max_client_batch_size
            );
            tracing::warn!("{message}");
            let err = ErrorResponse {
                error: message,
                error_type: ErrorType::Validation,
            };
            let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "batch_size");
            counter.increment(1);
            Err(err)?;
        }

        let batch_counter = if batch_size == 1 {
            None
        } else {
            Some(Arc::new(AtomicUsize::new(batch_size)))
        };
        let mut futures = Vec::with_capacity(batch_size);
        let mut compute_chars = 0;

        for input in &req.inputs {
            compute_chars += input.len();
            let local_infer = infer.clone();
            let local_batch_counter = batch_counter.clone();
            futures.push(async move {
                let permit = local_infer.acquire_permit().await;
                local_infer
                    .predict_tokens(
                        input.clone(),
                        truncate,
                        req.truncation_direction.into(),
                        req.raw_scores,
                        permit,
                        local_batch_counter,
                    )
                    .await
            });
        }
        let results = join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, TextEmbeddingsError>>()
            .map_err(ErrorResponse::from)?;

        let id2label = match &info.model_type {
            ModelType::Classifier(classifier) => &classifier.id2label,
            ModelType::Reranker(classifier) => &classifier.id2label,
            _ => panic!(),
        };

        let mut predictions = super::result_storage::vector(batch_size)
            .map_err(TextEmbeddingsError::from)
            .map_err(ErrorResponse::from)?;
        let mut total_tokenization_time = 0;
        let mut total_queue_time = 0;
        let mut total_inference_time = 0;
        let mut total_compute_tokens = 0;

        for response in results {
            total_compute_tokens += response.metadata.prompt_tokens;
            total_tokenization_time += response.metadata.tokenization.as_nanos() as u64;
            total_queue_time += response.metadata.queue.as_nanos() as u64;
            total_inference_time += response.metadata.inference.as_nanos() as u64;
            let token_predictions = response.results;

            let mut token_predictions_result =
                super::result_storage::vector(token_predictions.len())
                    .map_err(TextEmbeddingsError::from)
                    .map_err(ErrorResponse::from)?;
            for (token, token_id, scores, start, end) in token_predictions {
                // Check for NaN scores
                if scores.iter().any(|s| s.is_nan()) {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: "score is NaN".to_string(),
                            error_type: ErrorType::Backend,
                        }),
                    ));
                }

                // Sort aliases by label and original class index without allocating
                // sort scratch; retain the last class score for duplicate labels.
                let mut indexed =
                    super::result_storage::vector::<(String, f32, usize)>(scores.len())
                        .map_err(TextEmbeddingsError::from)
                        .map_err(ErrorResponse::from)?;
                for (index, score) in scores.into_iter().enumerate() {
                    let key = super::result_storage::index(index)
                        .map_err(TextEmbeddingsError::from)
                        .map_err(ErrorResponse::from)?;
                    let label = id2label
                        .get(&key)
                        .expect("id2label missing label for score index. This is a bug.");
                    indexed.push((
                        super::result_storage::string(label)
                            .map_err(TextEmbeddingsError::from)
                            .map_err(ErrorResponse::from)?,
                        score,
                        index,
                    ));
                }
                indexed.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));
                indexed.dedup_by(|later, earlier| {
                    if later.0 == earlier.0 {
                        earlier.1 = later.1;
                        earlier.2 = later.2;
                        true
                    } else {
                        false
                    }
                });
                let mut results = super::result_storage::vector(indexed.len())
                    .map_err(TextEmbeddingsError::from)
                    .map_err(ErrorResponse::from)?;
                results.extend(indexed.into_iter().map(|(label, score, _)| (label, score)));

                token_predictions_result.push(crate::http::types::TokenPrediction {
                    token,
                    token_id,
                    start,
                    end,
                    results,
                });
            }
            predictions.push(token_predictions_result);
        }
        let batch_size = batch_size as u64;

        // Apply aggregation strategy
        for (batch_predictions, sentence) in predictions.iter_mut().zip(&req.inputs) {
            *batch_predictions = apply_aggregation(
                sentence,
                std::mem::take(batch_predictions),
                &req.aggregation_strategy,
                id2label,
                &req.ignore_labels,
            )
            .map_err(TextEmbeddingsError::from)
            .map_err(ErrorResponse::from)?;
        }

        let counter = metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "batch");
        counter.increment(1);

        (
            TokenPredictResponse::Batch(predictions),
            ResponseMetadata::new(
                compute_chars,
                total_compute_tokens,
                start_time,
                Duration::from_nanos(total_tokenization_time / batch_size),
                Duration::from_nanos(total_queue_time / batch_size),
                Duration::from_nanos(total_inference_time / batch_size),
                service_tier,
            ),
        )
    };

    metadata.record_span(&span);
    metadata.record_metrics(&info.model_id);

    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    Ok((headers, Json(response)))
}

/// Get Ranks. Returns a 424 status code if the model is not a Sequence Classification model with
/// a single class.
#[utoipa::path(
post,
tag = "infinity_v3",
path = "/rerank",
request_body = RerankRequest,
responses(
(status = 200, description = "Ranks", body = RerankResponse),
(status = 424, description = "Rerank Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn rerank(
    context: Extension<Option<opentelemetry::Context>>,
    RoutedJson { runtime, value }: RoutedJson<RerankRequest>,
) -> Response {
    rerank_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        context,
        Json(value),
    )
    .await
    .into_response()
}

async fn rerank_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Extension(context): Extension<Option<opentelemetry::Context>>,
    Json(req): Json<RerankRequest>,
) -> Result<(HeaderMap, Json<RerankResponse>), (StatusCode, Json<ErrorResponse>)> {
    let infer = Extension(infer.0.with_service_tier(req.service_tier));
    let service_tier = infer.service_tier();
    let span = tracing::Span::current();
    if let Some(context) = context {
        span.set_parent(context);
    }

    let start_time = Instant::now();

    if req.texts.is_empty() {
        let message = "`texts` cannot be empty".to_string();
        tracing::warn!("{message}");
        let err = ErrorResponse {
            error: message,
            error_type: ErrorType::Empty,
        };
        let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "validation");
        counter.increment(1);
        Err(err)?;
    }

    match &info.model_type {
        ModelType::Reranker(_) => Ok(()),
        ModelType::Decision | ModelType::Classifier(_) | ModelType::Embedding(_) => {
            let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "model_type");
            counter.increment(1);
            let message = "model is not a re-ranker model".to_string();
            Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )))
        }
    }
    .map_err(|err| {
        tracing::warn!("{err}");
        ErrorResponse::from(err)
    })?;

    // Closure for rerank
    let rerank_inner = move |query: String,
                             text: String,
                             truncate: bool,
                             infer: Infer,
                             batch_counter: Option<Arc<AtomicUsize>>| async move {
        let permit = infer.acquire_permit().await;

        let response = infer
            .predict(
                (query, text),
                truncate,
                req.truncation_direction.into(),
                req.raw_scores,
                permit,
                batch_counter,
            )
            .await
            .map_err(ErrorResponse::from)?;

        let score = response.results[0];

        Ok::<(usize, Duration, Duration, Duration, f32), ErrorResponse>((
            response.metadata.prompt_tokens,
            response.metadata.tokenization,
            response.metadata.queue,
            response.metadata.inference,
            score,
        ))
    };

    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = {
        let counter = metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "batch");
        counter.increment(1);

        let batch_size = req.texts.len();
        if batch_size > info.max_client_batch_size {
            let message = format!(
                "batch size {batch_size} > maximum allowed batch size {}",
                info.max_client_batch_size
            );
            tracing::warn!("{message}");
            let err = ErrorResponse {
                error: message,
                error_type: ErrorType::Validation,
            };
            let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "batch_size");
            counter.increment(1);
            Err(err)?;
        }

        let batch_counter = if batch_size == 1 {
            None
        } else {
            Some(Arc::new(AtomicUsize::new(batch_size)))
        };
        let mut futures = Vec::with_capacity(batch_size);
        let query_chars = req.query.chars().count();
        let mut compute_chars = query_chars * batch_size;

        for text in &req.texts {
            compute_chars += text.chars().count();
            let local_infer = infer.clone();
            let local_batch_counter = batch_counter.clone();
            futures.push(rerank_inner(
                req.query.clone(),
                text.clone(),
                truncate,
                local_infer.0,
                local_batch_counter,
            ))
        }
        let results = join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<(usize, Duration, Duration, Duration, f32)>, ErrorResponse>>()?;

        let mut ranks = Vec::with_capacity(batch_size);
        let mut total_tokenization_time = 0;
        let mut total_queue_time = 0;
        let mut total_inference_time = 0;
        let mut total_compute_tokens = 0;

        for (index, (r, original_text)) in results.into_iter().zip(req.texts).enumerate() {
            total_compute_tokens += r.0;
            total_tokenization_time += r.1.as_nanos() as u64;
            total_queue_time += r.2.as_nanos() as u64;
            total_inference_time += r.3.as_nanos() as u64;
            let text = if req.return_text {
                Some(original_text)
            } else {
                None
            };

            let score = r.4;
            // Check that s is not NaN or the partial_cmp below will panic
            if score.is_nan() {
                Err(ErrorResponse {
                    error: "score is NaN".to_string(),
                    error_type: ErrorType::Backend,
                })?;
            }

            ranks.push(Rank { index, text, score })
        }

        // Reverse sort
        ranks.sort_by(|x, y| x.score.partial_cmp(&y.score).unwrap());
        ranks.reverse();

        let batch_size = batch_size as u64;

        let counter = metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "batch");
        counter.increment(1);

        (
            RerankResponse(ranks),
            ResponseMetadata::new(
                compute_chars,
                total_compute_tokens,
                start_time,
                Duration::from_nanos(total_tokenization_time / batch_size),
                Duration::from_nanos(total_queue_time / batch_size),
                Duration::from_nanos(total_inference_time / batch_size),
                service_tier,
            ),
        )
    };

    metadata.record_span(&span);
    metadata.record_metrics(&info.model_id);

    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    Ok((headers, Json(response)))
}

/// Get Sentence Similarity. Returns a 424 status code if the model is not an embedding model.
#[utoipa::path(
post,
tag = "infinity_v3",
path = "/similarity",
request_body = SimilarityRequest,
responses(
(status = 200, description = "Sentence Similarity", body = SimilarityResponse),
(status = 424, description = "Embedding Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn similarity(
    context: Extension<Option<opentelemetry::Context>>,
    RoutedJson { runtime, value }: RoutedJson<SimilarityRequest>,
) -> Response {
    similarity_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        context,
        Json(value),
    )
    .await
    .into_response()
}

async fn similarity_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    context: Extension<Option<opentelemetry::Context>>,
    Json(req): Json<SimilarityRequest>,
) -> Result<(HeaderMap, Json<SimilarityResponse>), (StatusCode, Json<ErrorResponse>)> {
    if info.is_multi_vector() {
        return Err(ErrorResponse {
            error: "Multi-vector embeddings require late-interaction scoring".into(),
            error_type: ErrorType::Validation,
        }
        .into());
    }
    if req.inputs.sentences.is_empty() {
        let message = "`inputs.sentences` cannot be empty".to_string();
        tracing::warn!("{message}");
        let err = ErrorResponse {
            error: message,
            error_type: ErrorType::Empty,
        };
        let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "validation");
        counter.increment(1);
        Err(err)?;
    }
    // +1 because of the source sentence
    let batch_size = req.inputs.sentences.len() + 1;
    if batch_size > info.max_client_batch_size {
        let message = format!(
            "batch size {batch_size} > maximum allowed batch size {}",
            info.max_client_batch_size
        );
        tracing::warn!("{message}");
        let err = ErrorResponse {
            error: message,
            error_type: ErrorType::Validation,
        };
        let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "batch_size");
        counter.increment(1);
        Err(err)?;
    }

    // Convert request to embed request
    let mut inputs = Vec::with_capacity(req.inputs.sentences.len() + 1);
    inputs.push(req.inputs.source_sentence);
    for s in req.inputs.sentences {
        inputs.push(s);
    }
    let parameters = req.parameters.unwrap_or_default();
    let embed_req = EmbedRequest {
        service_tier: req.service_tier,
        encoding_format: EncodingFormat::Float,
        model: None,
        inputs: EmbeddingInput::TextBatch(inputs),
        truncate: parameters.truncate,
        truncation_direction: parameters.truncation_direction,
        prompt_name: parameters.prompt_name,
        normalize: false,
        dimensions: None,
    };

    // Get embeddings
    let (header_map, embed_response) = embed_inner(infer, info, context, Json(embed_req)).await?;
    let embeddings = embed_response.0 .0;

    // Compute cosine
    reserve_result_allocation::<f32>(batch_size - 1).map_err(ErrorResponse::from)?;
    let Embedding::Float(source) = &embeddings[0] else {
        unreachable!("similarity requests float embeddings");
    };
    let distances = (1..batch_size)
        .map(|i| {
            let Embedding::Float(candidate) = &embeddings[i] else {
                unreachable!("similarity requests float embeddings");
            };
            1.0 - f32::cosine(source, candidate).unwrap() as f32
        })
        .collect();

    Ok((header_map, Json(SimilarityResponse(distances))))
}

/// Get Embeddings. Returns a 424 status code if the model is not an embedding model.
#[utoipa::path(
post,
tag = "infinity_v3",
path = "/embed",
request_body = EmbedRequest,
responses(
(status = 200, description = "Embeddings", body = EmbedResponse),
(status = 424, description = "Embedding Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn embed(
    context: Extension<Option<opentelemetry::Context>>,
    RoutedJson { runtime, value }: RoutedJson<EmbedRequest>,
) -> Response {
    embed_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        context,
        Json(value),
    )
    .await
    .into_response()
}

async fn embed_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Extension(context): Extension<Option<opentelemetry::Context>>,
    Json(req): Json<EmbedRequest>,
) -> Result<(HeaderMap, Json<EmbedResponse>), (StatusCode, Json<ErrorResponse>)> {
    let infer = Extension(infer.0.with_service_tier(req.service_tier));
    let service_tier = infer.service_tier();
    if info.is_multi_vector() {
        if req.dimensions.is_some() {
            return Err(ErrorResponse {
                error: "Multi-vector embeddings do not support dimensions".into(),
                error_type: ErrorType::Validation,
            }
            .into());
        }
        let (headers, Json(values)) = embed_all_inner(
            infer,
            info,
            Extension(context),
            Json(EmbedAllRequest {
                service_tier: req.service_tier,
                encoding_format: req.encoding_format,
                model: req.model,
                inputs: req.inputs,
                truncate: req.truncate,
                truncation_direction: req.truncation_direction,
                prompt_name: req.prompt_name,
            }),
        )
        .await?;
        return Ok((
            headers,
            Json(EmbedResponse(
                values.0.into_iter().map(Embedding::All).collect(),
            )),
        ));
    }
    let span = tracing::Span::current();
    if let Some(context) = context {
        span.set_parent(context);
    }

    let start_time = Instant::now();

    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = match InputBatch::from(req.inputs) {
        InputBatch::Single(input) => {
            metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "single").increment(1);

            let compute_chars = input.count_chars();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_pooled(
                    input,
                    truncate,
                    req.truncation_direction.into(),
                    req.prompt_name,
                    req.normalize,
                    req.dimensions,
                    permit,
                    None,
                )
                .await
                .map_err(ErrorResponse::from)?;

            metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "single").increment(1);

            (
                vec![response.results],
                ResponseMetadata::new(
                    compute_chars,
                    response.metadata.prompt_tokens,
                    start_time,
                    response.metadata.tokenization,
                    response.metadata.queue,
                    response.metadata.inference,
                    service_tier,
                ),
            )
        }
        InputBatch::Batch(inputs) => {
            let counter = metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "batch");
            counter.increment(1);

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "validation");
                counter.increment(1);
                Err(err)?;
            }

            let batch_size = inputs.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let batch_counter = if batch_size == 1 {
                None
            } else {
                Some(Arc::new(AtomicUsize::new(batch_size)))
            };
            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.count_chars();

                let local_infer = infer.clone();
                let prompt_name = req.prompt_name.clone();
                let local_batch_counter = batch_counter.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    local_infer
                        .embed_pooled(
                            input,
                            truncate,
                            req.truncation_direction.into(),
                            prompt_name,
                            req.normalize,
                            req.dimensions,
                            permit,
                            local_batch_counter,
                        )
                        .await
                })
            }
            let results = join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<PooledEmbeddingsInferResponse>, TextEmbeddingsError>>()
                .map_err(ErrorResponse::from)?;

            let mut embeddings = Vec::with_capacity(batch_size);
            let mut total_tokenization_time = 0;
            let mut total_queue_time = 0;
            let mut total_inference_time = 0;
            let mut total_compute_tokens = 0;

            for r in results {
                total_tokenization_time += r.metadata.tokenization.as_nanos() as u64;
                total_queue_time += r.metadata.queue.as_nanos() as u64;
                total_inference_time += r.metadata.inference.as_nanos() as u64;
                total_compute_tokens += r.metadata.prompt_tokens;
                embeddings.push(r.results);
            }
            let batch_size = batch_size as u64;

            let counter = metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "batch");
            counter.increment(1);

            (
                embeddings,
                ResponseMetadata::new(
                    compute_chars,
                    total_compute_tokens,
                    start_time,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                    service_tier,
                ),
            )
        }
    };

    let mut encoded = super::result_storage::vector(response.len())
        .map_err(|err| ErrorResponse::from(TextEmbeddingsError::from(err)))?;
    for values in response {
        encoded.push(encode_vector(values, req.encoding_format).map_err(ErrorResponse::from)?);
    }
    let response = EmbedResponse(encoded);

    metadata.record_span(&span);
    metadata.record_metrics(&info.model_id);

    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    Ok((headers, Json(response)))
}

/// Get Sparse Embeddings. Returns a 424 status code if the model is not an embedding model with SPLADE pooling.
#[utoipa::path(
post,
tag = "infinity_v3",
path = "/embed_sparse",
request_body = EmbedSparseRequest,
responses(
(status = 200, description = "Embeddings", body = EmbedSparseResponse),
(status = 424, description = "Embedding Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn embed_sparse(
    context: Extension<Option<opentelemetry::Context>>,
    RoutedJson { runtime, value }: RoutedJson<EmbedSparseRequest>,
) -> Response {
    embed_sparse_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        context,
        Json(value),
    )
    .await
    .into_response()
}

async fn embed_sparse_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Extension(context): Extension<Option<opentelemetry::Context>>,
    Json(req): Json<EmbedSparseRequest>,
) -> Result<(HeaderMap, Json<EmbedSparseResponse>), (StatusCode, Json<ErrorResponse>)> {
    let infer = Extension(infer.0.with_service_tier(req.service_tier));
    let service_tier = infer.service_tier();
    let span = tracing::Span::current();
    if let Some(context) = context {
        span.set_parent(context);
    }

    let start_time = Instant::now();

    let sparsify = |values: Vec<f32>| -> Result<Vec<SparseValue>, TextEmbeddingsError> {
        reserve_result_allocation::<SparseValue>(values.len())?;
        let mut sparse_values = Vec::with_capacity(values.len());
        for (index, value) in values.into_iter().enumerate() {
            if value != 0.0 {
                sparse_values.push(SparseValue { index, value });
            }
        }
        Ok(sparse_values)
    };
    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = match InputBatch::from(req.inputs) {
        InputBatch::Single(input) => {
            metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "single").increment(1);

            let compute_chars = input.count_chars();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_sparse(
                    input,
                    truncate,
                    req.truncation_direction.into(),
                    req.prompt_name,
                    permit,
                    None,
                )
                .await
                .map_err(ErrorResponse::from)?;

            metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "single").increment(1);

            (
                EmbedSparseResponse(vec![
                    sparsify(response.results).map_err(ErrorResponse::from)?
                ]),
                ResponseMetadata::new(
                    compute_chars,
                    response.metadata.prompt_tokens,
                    start_time,
                    response.metadata.tokenization,
                    response.metadata.queue,
                    response.metadata.inference,
                    service_tier,
                ),
            )
        }
        InputBatch::Batch(inputs) => {
            let counter = metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "batch");
            counter.increment(1);

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "validation");
                counter.increment(1);
                Err(err)?;
            }

            let batch_size = inputs.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let batch_counter = if batch_size == 1 {
                None
            } else {
                Some(Arc::new(AtomicUsize::new(batch_size)))
            };
            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.count_chars();

                let local_infer = infer.clone();
                let prompt_name = req.prompt_name.clone();
                let local_batch_counter = batch_counter.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    let response = local_infer
                        .embed_sparse(
                            input,
                            truncate,
                            req.truncation_direction.into(),
                            prompt_name,
                            permit,
                            local_batch_counter,
                        )
                        .await?;
                    Ok((sparsify(response.results)?, response.metadata))
                })
            }
            let results = join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<(Vec<SparseValue>, InferMetadata)>, TextEmbeddingsError>>()
                .map_err(ErrorResponse::from)?;

            let mut embeddings = Vec::with_capacity(batch_size);
            let mut total_tokenization_time = 0;
            let mut total_queue_time = 0;
            let mut total_inference_time = 0;
            let mut total_compute_tokens = 0;

            for r in results {
                total_tokenization_time += r.1.tokenization.as_nanos() as u64;
                total_queue_time += r.1.queue.as_nanos() as u64;
                total_inference_time += r.1.inference.as_nanos() as u64;
                total_compute_tokens += r.1.prompt_tokens;
                embeddings.push(r.0);
            }
            let batch_size = batch_size as u64;

            let counter = metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "batch");
            counter.increment(1);

            (
                EmbedSparseResponse(embeddings),
                ResponseMetadata::new(
                    compute_chars,
                    total_compute_tokens,
                    start_time,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                    service_tier,
                ),
            )
        }
    };

    metadata.record_span(&span);
    metadata.record_metrics(&info.model_id);

    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    Ok((headers, Json(response)))
}

/// Get all Embeddings without Pooling.
/// Returns a 424 status code if the model is not an embedding model.
#[utoipa::path(
post,
tag = "infinity_v3",
path = "/embed_all",
request_body = EmbedAllRequest,
responses(
(status = 200, description = "Embeddings", body = EmbedAllResponse),
(status = 424, description = "Embedding Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn embed_all(
    context: Extension<Option<opentelemetry::Context>>,
    RoutedJson { runtime, value }: RoutedJson<EmbedAllRequest>,
) -> Response {
    embed_all_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        context,
        Json(value),
    )
    .await
    .into_response()
}

async fn embed_all_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Extension(context): Extension<Option<opentelemetry::Context>>,
    Json(req): Json<EmbedAllRequest>,
) -> Result<(HeaderMap, Json<EmbedAllResponse>), (StatusCode, Json<ErrorResponse>)> {
    let infer = Extension(infer.0.with_service_tier(req.service_tier));
    let service_tier = infer.service_tier();
    let span = tracing::Span::current();
    if let Some(context) = context {
        span.set_parent(context);
    }

    let start_time = Instant::now();

    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = match InputBatch::from(req.inputs) {
        InputBatch::Single(input) => {
            metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "single").increment(1);

            let compute_chars = input.count_chars();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_all(
                    input,
                    truncate,
                    req.truncation_direction.into(),
                    req.prompt_name,
                    permit,
                    None,
                )
                .await
                .map_err(ErrorResponse::from)?;

            metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "single").increment(1);

            (
                vec![response.results],
                ResponseMetadata::new(
                    compute_chars,
                    response.metadata.prompt_tokens,
                    start_time,
                    response.metadata.tokenization,
                    response.metadata.queue,
                    response.metadata.inference,
                    service_tier,
                ),
            )
        }
        InputBatch::Batch(inputs) => {
            let counter = metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "batch");
            counter.increment(1);

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "validation");
                counter.increment(1);
                Err(err)?;
            }

            let batch_size = inputs.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let batch_counter = if batch_size == 1 {
                None
            } else {
                Some(Arc::new(AtomicUsize::new(batch_size)))
            };
            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.count_chars();

                let local_infer = infer.clone();
                let prompt_name = req.prompt_name.clone();
                let local_batch_counter = batch_counter.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    local_infer
                        .embed_all(
                            input,
                            truncate,
                            req.truncation_direction.into(),
                            prompt_name,
                            permit,
                            local_batch_counter,
                        )
                        .await
                })
            }
            let results = join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<AllEmbeddingsInferResponse>, TextEmbeddingsError>>()
                .map_err(ErrorResponse::from)?;

            let mut embeddings = Vec::with_capacity(batch_size);
            let mut total_tokenization_time = 0;
            let mut total_queue_time = 0;
            let mut total_inference_time = 0;
            let mut total_compute_tokens = 0;

            for r in results {
                total_tokenization_time += r.metadata.tokenization.as_nanos() as u64;
                total_queue_time += r.metadata.queue.as_nanos() as u64;
                total_inference_time += r.metadata.inference.as_nanos() as u64;
                total_compute_tokens += r.metadata.prompt_tokens;
                embeddings.push(r.results);
            }
            let batch_size = batch_size as u64;

            let counter = metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "batch");
            counter.increment(1);

            (
                embeddings,
                ResponseMetadata::new(
                    compute_chars,
                    total_compute_tokens,
                    start_time,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                    service_tier,
                ),
            )
        }
    };

    let mut encoded = super::result_storage::vector(response.len())
        .map_err(|err| ErrorResponse::from(TextEmbeddingsError::from(err)))?;
    for values in response {
        encoded.push(encode_matrix(values, req.encoding_format).map_err(ErrorResponse::from)?);
    }
    let response = EmbedAllResponse(encoded);

    metadata.record_span(&span);
    metadata.record_metrics(&info.model_id);

    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    Ok((headers, Json(response)))
}

/// OpenAI compatible route. Returns a 424 status code if the model is not an embedding model.
#[utoipa::path(
post,
tag = "infinity_v3",
path = "/v1/embeddings",
request_body = OpenAICompatRequest,
responses(
(status = 200, description = "Embeddings", body = OpenAICompatResponse),
(status = 424, description = "Embedding Error", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Inference failed", "type": "backend"})),
(status = 429, description = "Model is overloaded", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Model is overloaded", "type": "overloaded"})),
(status = 422, description = "Tokenization error", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Tokenization error", "type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Batch is empty", "type": "empty"})),
(status = 413, description = "Batch size error", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Batch size error", "type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn openai_embed(
    context: Extension<Option<opentelemetry::Context>>,
    RoutedJson { runtime, value }: RoutedJson<OpenAICompatRequest>,
) -> Response {
    openai_embed_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        context,
        Json(value),
    )
    .await
    .into_response()
}

async fn openai_embed_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Extension(context): Extension<Option<opentelemetry::Context>>,
    Json(req): Json<OpenAICompatRequest>,
) -> Result<(HeaderMap, Json<OpenAICompatResponse>), (StatusCode, Json<OpenAICompatErrorResponse>)>
{
    let infer = Extension(infer.0.with_service_tier(req.service_tier));
    let service_tier = infer.service_tier();
    if info.is_multi_vector() {
        let model = info.model_id.clone();
        let (headers, Json(values)) = embed_inner(
            infer,
            info,
            Extension(context),
            Json(EmbedRequest {
                service_tier: req.service_tier,
                encoding_format: req.encoding_format,
                dimensions: req.dimensions,
                normalize: true,
                model: req.model,
                inputs: req.input,
                truncate: None,
                truncation_direction: TruncationDirection::Right,
                prompt_name: None,
            }),
        )
        .await
        .map_err(|(status, Json(error))| (status, Json(OpenAICompatErrorResponse::from(error))))?;
        let tokens = headers["x-compute-tokens"]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        let data = values
            .0
            .into_iter()
            .enumerate()
            .map(|(index, value)| OpenAICompatEmbedding {
                object: "embedding",
                embedding: value,
                index,
            })
            .collect();
        return Ok((
            headers,
            Json(OpenAICompatResponse {
                object: "list",
                data,
                model,
                usage: OpenAICompatUsage {
                    prompt_tokens: tokens,
                    total_tokens: tokens,
                },
            }),
        ));
    }
    let encode_embedding =
        |array: Vec<f32>| encode_vector(array, req.encoding_format).map_err(ErrorResponse::from);

    let span = tracing::Span::current();
    if let Some(context) = context {
        span.set_parent(context);
    }

    let start_time = Instant::now();

    let truncate = info.auto_truncate;

    let (embeddings, metadata) = match InputBatch::from(req.input) {
        InputBatch::Single(input) => {
            metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "single").increment(1);

            let compute_chars = input.count_chars();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_pooled(
                    input,
                    truncate,
                    tokenizers::TruncationDirection::Right,
                    None,
                    true,
                    req.dimensions,
                    permit,
                    None,
                )
                .await
                .map_err(ErrorResponse::from)?;

            metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "single").increment(1);

            let embedding = encode_embedding(response.results)?;
            (
                vec![OpenAICompatEmbedding {
                    object: "embedding",
                    embedding,
                    index: 0,
                }],
                ResponseMetadata::new(
                    compute_chars,
                    response.metadata.prompt_tokens,
                    start_time,
                    response.metadata.tokenization,
                    response.metadata.queue,
                    response.metadata.inference,
                    service_tier,
                ),
            )
        }
        InputBatch::Batch(inputs) => {
            let counter = metrics::counter!("infinity_request_count", "model" => info.model_id.clone(), "method" => "batch");
            counter.increment(1);

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "validation");
                counter.increment(1);
                Err(err)?;
            }

            let batch_size = inputs.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let batch_counter = if batch_size == 1 {
                None
            } else {
                Some(Arc::new(AtomicUsize::new(batch_size)))
            };
            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.count_chars();

                let local_infer = infer.clone();
                let local_batch_counter = batch_counter.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    local_infer
                        .embed_pooled(
                            input,
                            truncate,
                            tokenizers::TruncationDirection::Right,
                            None,
                            true,
                            req.dimensions,
                            permit,
                            local_batch_counter,
                        )
                        .await
                })
            }
            let results = join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<PooledEmbeddingsInferResponse>, TextEmbeddingsError>>()
                .map_err(ErrorResponse::from)?;

            let mut embeddings = Vec::with_capacity(batch_size);
            let mut total_tokenization_time = 0;
            let mut total_queue_time = 0;
            let mut total_inference_time = 0;
            let mut total_compute_tokens = 0;

            for (i, r) in results.into_iter().enumerate() {
                total_tokenization_time += r.metadata.tokenization.as_nanos() as u64;
                total_queue_time += r.metadata.queue.as_nanos() as u64;
                total_inference_time += r.metadata.inference.as_nanos() as u64;
                total_compute_tokens += r.metadata.prompt_tokens;
                let embedding = encode_embedding(r.results)?;
                embeddings.push(OpenAICompatEmbedding {
                    object: "embedding",
                    embedding,
                    index: i,
                });
            }
            let batch_size = batch_size as u64;

            let counter = metrics::counter!("infinity_request_success", "model" => info.model_id.clone(), "method" => "batch");
            counter.increment(1);

            (
                embeddings,
                ResponseMetadata::new(
                    compute_chars,
                    total_compute_tokens,
                    start_time,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                    service_tier,
                ),
            )
        }
    };

    metadata.record_span(&span);
    metadata.record_metrics(&info.model_id);

    let compute_tokens = metadata.compute_tokens;
    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    let response = OpenAICompatResponse {
        object: "list",
        data: embeddings,
        model: info.model_id.clone(),
        usage: OpenAICompatUsage {
            prompt_tokens: compute_tokens,
            total_tokens: compute_tokens,
        },
    };
    Ok((headers, Json(response)))
}

/// Tokenize inputs
#[utoipa::path(
post,
tag = "infinity_v3",
path = "/tokenize",
request_body = TokenizeRequest,
responses(
(status = 200, description = "Tokenized ids", body = TokenizeResponse),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
)
)]
#[instrument(skip_all)]
async fn tokenize(RoutedJson { runtime, value }: RoutedJson<TokenizeRequest>) -> Response {
    tokenize_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        Json(value),
    )
    .await
    .map(|body| (crate::service_tier_headers(crate::QueueTier::Default), body))
    .into_response()
}

async fn tokenize_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let tokenize_inner = move |input: String,
                               add_special_tokens: bool,
                               prompt_name: Option<String>,
                               infer: Infer| async move {
        let (encoded_input, encoding) = infer
            .tokenize(input.clone(), add_special_tokens, prompt_name)
            .await
            .map_err(ErrorResponse::from)?;
        let input = encoded_input.unwrap_or(input);

        let tokens: Vec<SimpleToken> = into_tokens(encoding, &input)
            .into_iter()
            .map(|t| {
                let CoreSimpleToken {
                    id,
                    text,
                    special,
                    start,
                    stop,
                } = t;
                SimpleToken {
                    id,
                    text,
                    special,
                    start,
                    stop,
                }
            })
            .collect();
        Ok::<Vec<SimpleToken>, ErrorResponse>(tokens)
    };

    let tokens = match req.inputs {
        TokenizeInput::Single(input) => {
            vec![tokenize_inner(input, req.add_special_tokens, req.prompt_name, infer.0).await?]
        }
        TokenizeInput::Batch(inputs) => {
            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "validation");
                counter.increment(1);
                Err(err)?;
            }

            let batch_size = inputs.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            for input in inputs {
                futures.push(tokenize_inner(
                    input,
                    req.add_special_tokens,
                    req.prompt_name.clone(),
                    infer.0.clone(),
                ));
            }

            join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<Vec<SimpleToken>>, ErrorResponse>>()?
        }
    };
    Ok(Json(TokenizeResponse(tokens)))
}

/// Decode input ids
#[utoipa::path(
post,
tag = "infinity_v3",
path = "/decode",
request_body = DecodeRequest,
responses(
(status = 200, description = "Decoded ids", body = DecodeResponse),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
)
)]
#[instrument(skip_all)]
async fn decode(RoutedJson { runtime, value }: RoutedJson<DecodeRequest>) -> Response {
    decode_inner(
        Extension(runtime.infer.clone()),
        Extension(runtime.info.clone()),
        Json(value),
    )
    .await
    .map(|body| (crate::service_tier_headers(crate::QueueTier::Default), body))
    .into_response()
}

async fn decode_inner(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<DecodeRequest>,
) -> Result<Json<DecodeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let decode_inner = move |ids: Vec<u32>, skip_special_tokens: bool, infer: Infer| async move {
        let text = infer
            .decode(ids, skip_special_tokens)
            .await
            .map_err(ErrorResponse::from)?;
        Ok::<String, ErrorResponse>(text)
    };

    let texts = match req.ids {
        InputIds::Single(ids) => vec![decode_inner(ids, req.skip_special_tokens, infer.0).await?],
        InputIds::Batch(ids) => {
            if ids.is_empty() {
                let message = "`ids` cannot be empty".to_string();
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "validation");
                counter.increment(1);
                Err(err)?;
            }

            let batch_size = ids.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::warn!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                let counter = metrics::counter!("infinity_request_failure", "model" => info.model_id.clone(), "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            for ids in ids {
                futures.push(decode_inner(ids, req.skip_special_tokens, infer.0.clone()));
            }

            join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<String>, ErrorResponse>>()?
        }
    };
    Ok(Json(DecodeResponse(texts)))
}

/// Prometheus metrics scrape endpoint
#[utoipa::path(
get,
tag = "infinity_v3",
path = "/metrics",
responses((status = 200, description = "Prometheus Metrics", body = String))
)]
async fn metrics(prom_handle: Extension<PrometheusHandle>) -> String {
    prom_handle.render()
}

fn api_doc() -> utoipa::openapi::OpenApi {
    // OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
    paths(
    get_model_info,
    models,
    super::systemone::systemone,
    health,
    predict,
    classify,
    rerank,
    embed,
    embed_all,
    embed_sparse,
    openai_embed,
    similarity,
    tokenize,
    decode,
    metrics,
    ),
    components(
    schemas(
    ServiceTier,
    super::systemone::SystemOneRequest,
    super::systemone::SystemOneRequestBody,
    super::systemone::D3SystemOneRequest,
    super::systemone::PplxJointSystemOneRequest,
    super::systemone::SystemOneState,
    infinity_v3_core::input::ModelInput,
    infinity_v3_core::input::MessageInput,
    infinity_v3_core::input::Message,
    infinity_v3_core::input::MessageRole,
    infinity_v3_core::input::MessageContent,
    infinity_v3_core::input::ContentPart,
    infinity_v3_core::input::ImageSource,
    infinity_v3_core::input::ImageDetail,
    infinity_v3_core::input::AudioSource,
    infinity_v3_core::input::AudioFormat,
    infinity_v3_core::input::VideoSource,
    PredictInput,
    EmbeddingInput,
    Info,
    ModelType,
    ClassifierModel,
    Embedding,
    EncodingFormat,
    Tensor,
    super::types::AllEmbedding,
    EmbeddingModel,
    ClassifyRequest,
    ClassifyRequestId,
    ClassifyResponse,
    ClassifyData,
    ClassifyUsage,
    ClassifyErrorResponse,
    PredictRequest,
    Prediction,
    PredictResponse,
    OpenAICompatRequest,
    OpenAICompatEmbedding,
    OpenAICompatUsage,
    OpenAICompatResponse,
    EmbedAllRequest,
    EmbedAllResponse,
    EmbedSparseRequest,
    SparseValue,
    EmbedSparseResponse,
    RerankRequest,
    Rank,
    RerankResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    ServedModel,
    ServedModels,
    OpenAICompatErrorResponse,
    TokenizeInput,
    TokenizeRequest,
    TokenizeResponse,
    TruncationDirection,
    SimilarityInput,
    SimilarityParameters,
    SimilarityRequest,
    SimilarityResponse,
    SimpleToken,
    InputIds,
    DecodeRequest,
    DecodeResponse,
    ErrorType,
    )
    ),
    tags(
    (name = "infinity_v3", description = "infinity_v3 API")
    ),
    info(
    title = "infinity_v3",
    contact(name = "infinity authors"),
    license(
    name = "Infinity Restricted MIT License 1.0 (see scope and retained licenses)"
    )
    )
    )]
    struct ApiDoc;

    let mut doc = ApiDoc::openapi();
    if let Some(operation) = doc.paths.paths.get_mut("/predict").and_then(|path| {
        path.operations
            .get_mut(&utoipa::openapi::path::PathItemType::Post)
    }) {
        operation.deprecated = Some(utoipa::openapi::Deprecated::True);
    }

    // Optional wire fields are mutually exclusive in the actual request contract.
    if let Some(components) = doc.components.as_mut() {
        if let Some(schema) = components.schemas.get("ClassifyRequest").cloned() {
            components.schemas.insert(
                "ClassifyRequest".into(),
                super::types::exclusive_input_schema(schema),
            );
        }
    }
    super::response_format::document(&mut doc);
    doc
}

/// Serving method
#[allow(clippy::too_many_arguments)]
pub async fn run(
    registry: ModelRegistry,
    addr: SocketAddr,
    prom_builder: PrometheusBuilder,
    payload_limit: usize,
    api_key: Option<String>,
    cors_allow_origin: Option<Vec<String>>,
) -> Result<(), anyhow::Error> {
    // See: https://github.com/metrics-rs/metrics/issues/467#issuecomment-2022755151
    let (recorder, _) = prom_builder
        .build()
        .context("failed to build prometheus recorder")?;
    let prom_handle = recorder.handle();
    metrics::set_global_recorder(recorder).context("Failed to set global recorder")?;

    let app = build_router(
        registry,
        prom_handle,
        payload_limit,
        api_key,
        cors_allow_origin,
    )?;

    // Run server
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context(format!("Could not bind TCP Listener on {addr}"))?;

    tracing::info!("Starting HTTP server: {}", &addr);
    tracing::info!("Ready");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown::shutdown_signal())
        .await?;

    Ok(())
}

fn cors_layer(cors_allow_origin: Option<Vec<String>>) -> anyhow::Result<CorsLayer> {
    let billing = crate::billing_headers()?;
    // CORS allowed origins
    // map to go inside the option and then map to parse from String to HeaderValue
    // Finally, convert to AllowOrigin
    let allow_origin: Option<AllowOrigin> = cors_allow_origin.map(|cors_allow_origin| {
        if cors_allow_origin.iter().any(|origin| origin == "*") {
            AllowOrigin::any()
        } else {
            AllowOrigin::list(
                cors_allow_origin
                    .into_iter()
                    .map(|origin| origin.parse::<HeaderValue>().unwrap()),
            )
        }
    });

    // CORS layer
    let allow_origin = allow_origin.unwrap_or(AllowOrigin::any());
    Ok(CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE, AUTHORIZATION])
        .expose_headers(
            billing
                .iter()
                .map(|names| names.service_tier.clone())
                .collect::<Vec<_>>(),
        )
        .allow_origin(allow_origin))
}

fn build_router(
    registry: ModelRegistry,
    prom_handle: PrometheusHandle,
    payload_limit: usize,
    api_key: Option<String>,
    cors_allow_origin: Option<Vec<String>>,
) -> Result<Router, anyhow::Error> {
    let doc = api_doc();

    let cors_layer = cors_layer(cors_allow_origin)?;

    let mut routes = Router::new()
        // Base routes
        .route("/info", get(get_model_info))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/embed", post(embed))
        .route("/embed_all", post(embed_all))
        .route("/embed_sparse", post(embed_sparse))
        .route(
            "/predict",
            post(predict).layer(axum::middleware::map_response(deprecate_predict)),
        )
        .route("/predict_tokens", post(predict_tokens))
        .route("/rerank", post(rerank))
        .route("/similarity", post(similarity))
        .route("/tokenize", post(tokenize))
        .route("/decode", post(decode))
        // OpenAI compat route
        .route("/embeddings", post(openai_embed))
        .route("/v1/embeddings", post(openai_embed))
        .route("/v1/classify", post(classify))
        .route("/v1/systemone", post(super::systemone::systemone))
        .route("/", post(default_inference))
        .route("/invocations", post(default_inference));

    let public_routes = Router::new()
        // Base Health route
        .route("/health", get(health))
        // Inference API health route
        .route("/", get(health))
        // AWS Sagemaker health route
        .route("/ping", get(health))
        // Prometheus metrics route
        .route("/metrics", get(metrics));

    super::response_format::initialize_memory_errors();
    let http_output_limit =
        std::env::var(infinity_v3_config::infinity_env!("HTTP_OUTPUT_MAX_BYTES"))
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .with_context(|| {
                format!(
                    "Invalid {}",
                    infinity_v3_config::infinity_env!("HTTP_OUTPUT_MAX_BYTES")
                )
            })?
            .unwrap_or(512 * 1024 * 1024);
    let http_input_limit = std::env::var(infinity_v3_config::infinity_env!("HTTP_INPUT_MAX_BYTES"))
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .with_context(|| {
            format!(
                "Invalid {}",
                infinity_v3_config::infinity_env!("HTTP_INPUT_MAX_BYTES")
            )
        })?
        .unwrap_or(512 * 1024 * 1024);
    let admission = super::memory::MemoryAdmission::new(
        registry.models().iter().try_fold(0usize, |total, model| {
            total
                .checked_add(model.info.max_concurrent_requests)
                .context("Combined HTTP concurrency is unrepresentable")
        })?,
        payload_limit,
        http_input_limit,
        http_output_limit,
    )?;
    routes = routes.layer(axum::middleware::from_fn_with_state(
        admission.clone(),
        super::memory::admit,
    ));

    if let Some(api_key) = api_key {
        let prefix = format!("Bearer {}", api_key);

        // Leak to allow FnMut
        let api_key: &'static str = prefix.leak();

        let auth = move |headers: HeaderMap,
                         request: axum::extract::Request,
                         next: axum::middleware::Next| async move {
            match headers.get(AUTHORIZATION) {
                Some(token) if token == api_key => {
                    let response = next.run(request).await;
                    Ok(response)
                }
                _ => Err(StatusCode::UNAUTHORIZED),
            }
        };

        routes = routes.layer(axum::middleware::from_fn(auth));
    }

    Ok(Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", doc))
        .merge(routes)
        .merge(public_routes)
        .layer(Extension(registry))
        .layer(Extension(prom_handle.clone()))
        .layer(OtelAxumLayer::default())
        .layer(axum::middleware::from_fn(
            logging::http::trace_context_middleware,
        ))
        .layer(DefaultBodyLimit::max(payload_limit))
        .layer(axum::middleware::from_fn(super::response_format::negotiate))
        .layer(axum::middleware::from_fn_with_state(
            admission,
            super::memory::response_budget,
        ))
        .layer(axum::middleware::from_fn(
            super::response_format::retain_results,
        ))
        .layer(cors_layer))
}

impl From<&ErrorType> for StatusCode {
    fn from(value: &ErrorType) -> Self {
        match value {
            ErrorType::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
            ErrorType::Backend => StatusCode::FAILED_DEPENDENCY,
            ErrorType::Overloaded => StatusCode::TOO_MANY_REQUESTS,
            ErrorType::Tokenizer => StatusCode::UNPROCESSABLE_ENTITY,
            ErrorType::Validation => StatusCode::PAYLOAD_TOO_LARGE,
            ErrorType::Empty => StatusCode::BAD_REQUEST,
        }
    }
}

impl From<ErrorResponse> for OpenAICompatErrorResponse {
    fn from(value: ErrorResponse) -> Self {
        OpenAICompatErrorResponse {
            message: value.error,
            code: StatusCode::from(&value.error_type).as_u16(),
            error_type: value.error_type,
        }
    }
}

async fn default_inference(
    context: Extension<Option<opentelemetry::Context>>,
    RoutedJson { runtime, value }: RoutedJson<serde_json::Value>,
) -> Response {
    let infer = Extension(runtime.infer.clone());
    let info = Extension(runtime.info.clone());
    macro_rules! call {
        ($ty:ty, $handler:ident) => {
            match serde_json::from_value::<$ty>(value) {
                Ok(request) => $handler(infer, info, context, Json(request))
                    .await
                    .into_response(),
                Err(error) => (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(ErrorResponse {
                        error: error.to_string(),
                        error_type: ErrorType::Empty,
                    }),
                )
                    .into_response(),
            }
        };
    }
    match &runtime.info.model_type {
        ModelType::Decision => {
            super::systemone::systemone(super::model_routing::RoutedJson { runtime, value })
                .await
                .into_response()
        }
        ModelType::Classifier(_) => {
            let mut response = call!(PredictRequest, predict_inner);
            response = deprecate_predict(response).await;
            response
        }
        ModelType::Reranker(_) => call!(RerankRequest, rerank_inner),
        ModelType::Embedding(_)
            if std::env::var("TASK").ok().as_deref() == Some("sentence-similarity") =>
        {
            call!(SimilarityRequest, similarity_inner)
        }
        ModelType::Embedding(model) if model.pooling == "splade" => {
            call!(EmbedSparseRequest, embed_sparse_inner)
        }
        ModelType::Embedding(_) => call!(EmbedRequest, embed_inner),
    }
}

/// Convert to Axum supported formats
impl From<ErrorResponse> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: ErrorResponse) -> Self {
        (StatusCode::from(&err.error_type), Json(err))
    }
}

impl From<ErrorResponse> for (StatusCode, Json<OpenAICompatErrorResponse>) {
    fn from(err: ErrorResponse) -> Self {
        (StatusCode::from(&err.error_type), Json(err.into()))
    }
}

impl From<serde_json::Error> for ErrorResponse {
    fn from(err: serde_json::Error) -> Self {
        ErrorResponse {
            error: err.to_string(),
            error_type: ErrorType::Validation,
        }
    }
}

fn reserve_result_allocation<T>(count: usize) -> Result<(), TextEmbeddingsError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| BackendError::Inference("HTTP result size overflow".into()))?;
    infinity_v3_backend::result_memory::reserve_for_request(bytes)?;
    Ok(())
}

fn reserve_classification_output(
    scores: &[Vec<f32>],
    labels: &std::collections::HashMap<String, String>,
) -> Result<(), TextEmbeddingsError> {
    let overflow = || BackendError::Inference("Classification result size overflow".into());
    let mut bytes = scores
        .len()
        .checked_mul(std::mem::size_of::<Vec<Prediction>>() + std::mem::size_of::<ClassifyData>())
        .ok_or_else(overflow)?;
    for row in scores {
        bytes = bytes
            .checked_add(
                row.len()
                    .checked_mul(std::mem::size_of::<Prediction>())
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
        for index in 0..row.len() {
            let label_bytes = labels
                .get(&index.to_string())
                .map_or(6 + 20, |label| label.len());
            bytes = bytes.checked_add(label_bytes).ok_or_else(overflow)?;
        }
    }
    infinity_v3_backend::result_memory::reserve_for_request(bytes)?;
    Ok(())
}

#[cfg(test)]
mod classify_tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[tokio::test]
    async fn cors_preflight_allows_bearer_authorization() {
        use axum::{body::Body, http::Request};
        use tower::ServiceExt;

        for origins in [None, Some(vec!["https://client.example".into()])] {
            let app = Router::new()
                .route("/v1/classify", post(|| async { StatusCode::OK }))
                .layer(cors_layer(origins).unwrap());
            let response = app
                .oneshot(
                    Request::builder()
                        .method(Method::OPTIONS)
                        .uri("/v1/classify")
                        .header("origin", "https://client.example")
                        .header("access-control-request-method", "POST")
                        .header(
                            "access-control-request-headers",
                            "authorization,content-type",
                        )
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK);
            let headers = response.headers();
            let origin = headers["access-control-allow-origin"].to_str().unwrap();
            assert!(origin == "*" || origin == "https://client.example");
            let allowed = headers["access-control-allow-headers"]
                .to_str()
                .unwrap()
                .to_ascii_lowercase();
            let allowed: Vec<_> = allowed.split(',').map(str::trim).collect();
            assert!(allowed.contains(&"authorization"));
            assert!(allowed.contains(&"content-type"));
        }
    }

    #[test]
    fn sglang_response_preserves_class_order_batch_order_and_usage() {
        let labels = HashMap::from([
            ("0".into(), "negative".into()),
            ("1".into(), "positive".into()),
        ]);
        let response = classification_response(
            vec![vec![10000.0, 10002.0], vec![2.0, 2.0]],
            &labels,
            "model".into(),
            120,
            true,
        );
        assert!(response.id.starts_with("classify-"));
        assert!(uuid::Uuid::parse_str(response.id.strip_prefix("classify-").unwrap()).is_ok());
        assert!(response.created > 0);
        assert_eq!(response.data[0].label, "positive");
        assert_eq!(response.data[0].index, 0);
        assert_eq!(response.data[0].num_classes, 2);
        assert!((response.data[0].probs[0] - 0.11920292).abs() < 1e-6);
        assert!((response.data[0].probs[1] - 0.880_797_1).abs() < 1e-6);
        assert_eq!(response.data[1].label, "negative");
        assert_eq!(response.data[1].index, 1);
        assert_eq!(response.data[1].probs, vec![0.5, 0.5]);
        let value = serde_json::to_value(response).unwrap();
        assert_eq!(value["object"], "list");
        assert_eq!(value["model"], "model");
        assert_eq!(
            value["usage"],
            json!({"prompt_tokens":120,"total_tokens":120,"completion_tokens":0,"prompt_tokens_details":null})
        );
    }

    #[test]
    fn one_label_classifier_matches_sglang_softmax_and_labels_fall_back() {
        let data = classification_data(vec![vec![-20.0], vec![0.0, 1.0]], &HashMap::new(), true);
        assert_eq!(data[0].probs, vec![1.0]);
        assert_eq!(data[0].label, "LABEL_0");
        assert_eq!(data[1].label, "LABEL_1");
    }

    #[test]
    fn use_activation_false_returns_raw_logits() {
        let data = classification_data(vec![vec![-2.5], vec![0.0, 1.0]], &HashMap::new(), false);
        assert_eq!(data[0].probs, vec![-2.5]);
        assert_eq!(data[1].probs, vec![0.0, 1.0]);
        assert_eq!(data[1].label, "LABEL_1");
        let request: ClassifyRequest =
            serde_json::from_value(json!({"input": "x", "use_activation": false})).unwrap();
        assert_eq!(request.use_activation, Some(false));
    }

    #[test]
    fn classify_errors_have_sglang_wire_shape_and_status() {
        let (status, Json(error)) = classify_error(
            StatusCode::BAD_REQUEST,
            "bad input".into(),
            "invalid_request_error",
        );
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            serde_json::to_value(error).unwrap(),
            json!({"object":"error","message":"bad input","type":"invalid_request_error","param":null,"code":400})
        );
        let (status, Json(error)) = classify_inference_error(ErrorResponse {
            error: "busy".into(),
            error_type: ErrorType::Overloaded,
        });
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(error.code, 429);
    }

    #[tokio::test]
    async fn deprecation_preserves_success_and_error_bodies() {
        for status in [
            StatusCode::OK,
            StatusCode::UNPROCESSABLE_ENTITY,
            StatusCode::TOO_MANY_REQUESTS,
        ] {
            let response = axum::response::Response::builder()
                .status(status)
                .body(axum::body::Body::from("unchanged"))
                .unwrap();
            let response = deprecate_predict(response).await;
            assert_eq!(response.status(), status);
            assert_eq!(response.headers()["deprecation"], "@1791590400");
            assert_eq!(
                response.headers()["link"],
                "</v1/classify>; rel=\"successor-version\""
            );
            assert_eq!(
                axum::body::to_bytes(response.into_body(), 100)
                    .await
                    .unwrap(),
                "unchanged"
            );
        }
    }

    #[test]
    fn systemone_openapi_schema_snapshot() {
        let doc = serde_json::to_value(api_doc()).unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/openapi.json");
        let mut saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let names = [
            "ServiceTier",
            "EmbedRequest",
            "EmbedAllRequest",
            "EmbedSparseRequest",
            "OpenAICompatRequest",
            "PredictRequest",
            "ClassifyRequest",
            "RerankRequest",
            "SimilarityRequest",
            "TokenizeRequest",
            "DecodeRequest",
            "SystemOneRequestBody",
            "SystemOneRequest",
            "SystemOneState",
            "D3SystemOneRequest",
            "PplxJointSystemOneRequest",
        ];
        // UPDATE_OPENAPI=1 cargo test --lib systemone_openapi_schema_snapshot
        if std::env::var_os("UPDATE_OPENAPI").is_some() {
            saved["paths"]["/v1/systemone"] = doc["paths"]["/v1/systemone"].clone();
            for name in names {
                saved["components"]["schemas"][name] = doc["components"]["schemas"][name].clone();
            }
            std::fs::write(
                &path,
                format!("{}\n", serde_json::to_string_pretty(&saved).unwrap()),
            )
            .unwrap();
        }
        assert_eq!(
            doc["paths"]["/v1/systemone"],
            saved["paths"]["/v1/systemone"]
        );
        for name in names {
            assert_eq!(
                doc["components"]["schemas"][name], saved["components"]["schemas"][name],
                "{name}"
            );
        }
        for name in [
            "SystemOneRequest",
            "D3SystemOneRequest",
            "PplxJointSystemOneRequest",
        ] {
            assert_eq!(
                doc["components"]["schemas"][name]["properties"]["service_tier"]["$ref"],
                "#/components/schemas/ServiceTier"
            );
        }
        let operation = &doc["paths"]["/v1/systemone"]["post"];
        for status in ["400", "422", "429"] {
            for response in operation["responses"][status]["content"]
                .as_object()
                .unwrap()
                .values()
            {
                let reference = response["schema"]["$ref"].as_str().unwrap();
                assert_eq!(reference, "#/components/schemas/ErrorResponse");
                assert!(doc.pointer(reference.strip_prefix('#').unwrap()).is_some());
            }
        }
        assert_eq!(
            operation["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/SystemOneRequestBody"
        );
        assert!(doc["components"]["schemas"]["SystemOneRequestBody"]["anyOf"].is_array());
        assert!(doc["components"]["schemas"]["SystemOneRequest"]["properties"]["images"].is_null());
        assert_eq!(
            doc["components"]["schemas"]["D3SystemOneRequest"]["properties"]["images"]["items"]
                ["type"],
            "string"
        );
        assert_eq!(
            doc["components"]["schemas"]["PplxJointSystemOneRequest"]["properties"]["videos"]
                ["maxItems"],
            0
        );
    }

    #[test]
    fn classify_openapi_and_predict_deprecation_are_documented() {
        let doc = serde_json::to_value(api_doc()).unwrap();
        let checked_in: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/openapi.json")).unwrap();
        for (path, item) in doc["paths"].as_object().unwrap() {
            // Some existing runtime routes are absent from the checked-in reference.
            if !checked_in["paths"][path].is_null() {
                assert_eq!(item, &checked_in["paths"][path], "{path}");
            }
            for operation in item.as_object().unwrap().values() {
                for response in operation["responses"].as_object().unwrap().values() {
                    let content = &response["content"];
                    if content["application/json"]["schema"].get("$ref").is_some() {
                        let mut expected =
                            serde_json::from_value(content["application/json"]["schema"].clone())
                                .unwrap();
                        super::super::tensor::messagepack_schema(&mut expected);
                        assert_eq!(
                            serde_json::to_value(expected).unwrap(),
                            content["application/msgpack"]["schema"],
                            "{path}"
                        );
                    } else if !content["application/json"].is_null() {
                        assert_eq!(
                            content["application/json"]["schema"],
                            content["application/msgpack"]["schema"],
                            "{path}"
                        );
                    }
                }
            }
        }
        for schema in [
            "ServedModel",
            "ServedModels",
            "ClassifyRequest",
            "ClassifyRequestId",
            "ClassifyResponse",
            "ClassifyData",
            "ClassifyUsage",
            "ClassifyErrorResponse",
            "Embedding",
            "EncodingFormat",
            "Tensor",
            "AllEmbedding",
            "EmbedRequest",
            "EmbedAllRequest",
            "EmbedResponse",
            "EmbedAllResponse",
            "OpenAICompatRequest",
            "TensorMessagePack",
            "AllEmbeddingMessagePack",
            "EmbeddingMessagePack",
            "EmbedResponseMessagePack",
            "EmbedAllResponseMessagePack",
            "OpenAICompatEmbeddingMessagePack",
            "OpenAICompatResponseMessagePack",
        ] {
            assert!(doc["components"]["schemas"][schema].is_object(), "{schema}");
            assert_eq!(
                doc["components"]["schemas"][schema], checked_in["components"]["schemas"][schema],
                "{schema}"
            );
        }

        for (name, format, encoding) in [
            ("Tensor", "byte", "base64"),
            ("TensorMessagePack", "binary", "binary"),
        ] {
            let schema = &doc["components"]["schemas"][name];
            assert_eq!(schema, &checked_in["components"]["schemas"][name]);
            assert_eq!(schema["properties"]["data"]["format"], format);
            assert_eq!(
                schema["properties"]["data_encoding"]["enum"],
                serde_json::json!([encoding])
            );
        }
        let packed = &doc["paths"]["/v1/embeddings"]["post"]["responses"]["200"]["content"]
            ["application/msgpack"]["schema"]["$ref"];
        assert_eq!(
            packed,
            "#/components/schemas/OpenAICompatResponseMessagePack"
        );
        assert_eq!(doc["paths"]["/predict"]["post"]["deprecated"], true);
        assert_eq!(
            doc["paths"]["/v1/classify"]["post"]["responses"]["200"]["content"]["application/json"]
                ["schema"]["$ref"],
            "#/components/schemas/ClassifyResponse"
        );
    }
}

#[cfg(all(test, feature = "candle"))]
mod multi_model_test {
    use super::*;
    use crate::{initialize_model, ModelSettings};
    #[cfg(feature = "libtorch")]
    use axum::body::to_bytes;
    use axum::{body::Body, http::Request};
    use serde_json::json;
    #[cfg(feature = "libtorch")]
    use serde_json::Value;
    use std::sync::Arc;
    use tower::ServiceExt;

    fn settings(
        model_id: String,
        name: &str,
        backend_kind: infinity_v3_backend::BackendKind,
    ) -> ModelSettings {
        ModelSettings {
            model_id,
            served_model_name: Some(name.into()),
            revision: None,
            tokenization_workers: Some(1),
            tokenization_queue_capacity: std::num::NonZeroUsize::new(32).unwrap(),
            dtype: Some(infinity_v3_backend::DType::Float32),
            pooling: None,
            decision_protocol: None,
            priority_queue_bonus_ms: 1000,
            flex_queue_offset_ms: 10000,
            max_concurrent_requests: 4,
            max_batch_tokens: 128,
            max_sequence_length: None,
            max_batch_requests: Some(4),
            radix_mlp_threshold: 0.,
            enable_fp8_dynamic: false,
            max_client_batch_size: 8,
            max_decision_questions: 8,
            auto_truncate: false,
            default_prompt: None,
            default_prompt_name: None,
            dense_path: None,
            device_id: Some(0),
            backend_device_ids: None,
            backend_kind,
            torch_device: infinity_v3_backend::TorchDevice::Cpu,
            multimodal_config: Default::default(),
        }
    }
    #[tokio::test]
    #[ignore = "requires MULTI_MODEL_EMBEDDING_DIR (local CPU embedding checkpoint)"]
    async fn service_tier_headers_follow_effective_queue_policy() -> anyhow::Result<()> {
        for (bonus, offset) in [(1000, 1000), (0, 1000), (1000, 0), (0, 0)] {
            let mut config = settings(
                std::env::var("MULTI_MODEL_EMBEDDING_DIR")?,
                "embed",
                infinity_v3_backend::BackendKind::Candle,
            );
            config.priority_queue_bonus_ms = bonus;
            config.flex_queue_offset_ms = offset;
            config.max_sequence_length = std::num::NonZeroUsize::new(128);
            let runtime = Arc::new(initialize_model(config, None, None, &mut None).await?);
            let app = build_router(
                ModelRegistry::new(vec![runtime.clone()])?,
                PrometheusBuilder::new().build_recorder().handle(),
                2_000_000,
                None,
                None,
            )?;
            let removed = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/vertex")
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"instances":[{"inputs":"hello"}]}"#))?,
                )
                .await?;
            assert_eq!(removed.status(), StatusCode::NOT_FOUND);
            let (headers, _) = super::super::systemone::systemone_inner(
                Extension(runtime.infer.clone()),
                Extension(runtime.info.clone()),
                Extension(Some(Arc::new(super::super::systemone::SystemOne::Laya(
                    super::super::systemone::tests::service(),
                )))),
                Json(json!({"state":"hello", "questions":{}, "service_tier":"priority"})),
            )
            .await
            .map_err(|(status, _)| anyhow::anyhow!("Empty Laya request failed: {status}"))?;
            if let Some(names) = crate::configured_billing_headers() {
                assert_eq!(headers[&names.service_tier], "default");
                assert_eq!(headers[&names.input_tokens], "0");
            }
            for tier in [
                None,
                Some("default"),
                Some("auto"),
                Some("flex"),
                Some("priority"),
            ] {
                for (path, mut body) in [
                    ("/", json!({"inputs":"hello"})),
                    ("/invocations", json!({"inputs":"hello"})),
                    ("/embed", json!({"inputs":"hello"})),
                    ("/embed", json!({"inputs":["hello", "world"]})),
                    ("/v1/embeddings", json!({"input":["hello", "world"]})),
                    ("/tokenize", json!({"inputs":"hello"})),
                    ("/decode", json!({"ids":[101,7592,102]})),
                    (
                        "/similarity",
                        json!({"inputs":{"source_sentence":"hello", "sentences":["world"]}}),
                    ),
                ] {
                    if let Some(tier) = tier {
                        body["service_tier"] = json!(tier);
                    }
                    let preprocessing = path == "/tokenize" || path == "/decode";
                    let expected = match tier {
                        Some("priority") if bonus != 0 && !preprocessing => "priority",
                        Some("flex") if offset != 0 && !preprocessing => "flex",
                        _ => "default",
                    };
                    let response = app
                        .clone()
                        .oneshot(
                            Request::builder()
                                .method("POST")
                                .uri(path)
                                .header("content-type", "application/json")
                                .header("origin", "https://client.example")
                                .body(Body::from(body.to_string()))?,
                        )
                        .await?;
                    assert_eq!(response.status(), StatusCode::OK, "{path}: {body}");
                    if let Some(names) = crate::configured_billing_headers() {
                        assert_eq!(
                            response.headers()[&names.service_tier],
                            expected,
                            "{path}: {body}"
                        );
                        assert_eq!(
                            response.headers()["access-control-expose-headers"],
                            names.service_tier.as_str()
                        );
                        if !preprocessing {
                            let actual = response.headers()["x-compute-tokens"]
                                .to_str()?
                                .parse::<usize>()?;
                            let reported = response.headers()[&names.input_tokens]
                                .to_str()?
                                .parse::<usize>()?;
                            if expected != "priority" || names.priority_multiplier.is_none() {
                                assert_eq!(reported, actual, "{path}: {body}");
                            }
                            if path == "/v1/embeddings" {
                                let bytes =
                                    axum::body::to_bytes(response.into_body(), 1_000_000).await?;
                                let body: serde_json::Value = serde_json::from_slice(&bytes)?;
                                assert_eq!(body["usage"]["prompt_tokens"], actual);
                            }
                        }
                    } else {
                        assert!(!response.headers().contains_key("x-service-tier"));
                        assert!(!response
                            .headers()
                            .contains_key("access-control-expose-headers"));
                    }
                }
            }
            runtime.infer.shutdown().await;
        }
        Ok(())
    }

    #[cfg(feature = "libtorch")]
    async fn request(app: &Router, path: &str, value: Option<Value>) -> (StatusCode, Value) {
        let builder = Request::builder()
            .uri(path)
            .header("Accept", "application/json")
            .header("Authorization", "Bearer test");
        let request = match value {
            Some(value) => builder
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(value.to_string()))
                .unwrap(),
            None => builder.body(Body::empty()).unwrap(),
        };
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1_000_000).await.unwrap();
        (
            status,
            if bytes.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&bytes).unwrap()
            },
        )
    }

    #[cfg(feature = "libtorch")]
    #[tokio::test]
    #[ignore = "requires MULTI_MODEL_EMBEDDING_DIR and MULTI_MODEL_RERANK_DIR (CPU checkpoints with <=128 input tokens)"]
    async fn resident_models_route_isolate_and_shutdown() -> anyhow::Result<()> {
        use infinity_v3_backend::BackendKind;
        let a = settings(
            std::env::var("MULTI_MODEL_EMBEDDING_DIR")?,
            "embed",
            BackendKind::Candle,
        );
        let b = settings(
            std::env::var("MULTI_MODEL_RERANK_DIR")?,
            "rerank",
            BackendKind::Libtorch,
        );
        crate::model_registry::validate_configs(&[a.clone(), b.clone()])?;
        assert!(crate::model_registry::validate_configs(&[a.clone(), a.clone()]).is_err());
        let embedding = Arc::new(initialize_model(a, None, None, &mut None).await?);
        let reranker = Arc::new(initialize_model(b, None, None, &mut None).await?);
        let registry = ModelRegistry::new(vec![embedding.clone(), reranker.clone()])?;
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::set_global_recorder(recorder)?;
        let app = build_router(
            registry.clone(),
            handle.clone(),
            2_000_000,
            Some("test".into()),
            None,
        )?;
        let single_a = build_router(
            ModelRegistry::new(vec![embedding.clone()])?,
            handle.clone(),
            2_000_000,
            None,
            None,
        )?;
        let single_b = build_router(
            ModelRegistry::new(vec![reranker.clone()])?,
            handle.clone(),
            2_000_000,
            None,
            None,
        )?;
        let embedding_request =
            json!({"model":"embed", "input":["hello world", "another example"]});
        let rerank_request =
            json!({"model":"rerank", "query":"query", "texts":["good document", "bad document"]});
        assert_eq!(request(&app, "/health", None).await.0, StatusCode::OK);
        let catalog = request(&app, "/v1/models", None).await.1;
        assert_eq!(catalog["data"][0]["id"], "embed");
        assert_eq!(catalog["data"][0]["backend"], "candle");
        assert_eq!(catalog["data"][1]["id"], "rerank");
        assert_eq!(catalog["data"][1]["backend"], "libtorch");
        assert_eq!(
            request(&app, "/info?model=rerank", None).await.1["model_id"],
            "rerank"
        );
        assert_eq!(
            request(&app, "/info", None).await.0,
            StatusCode::BAD_REQUEST
        );
        let (actual_a, actual_b) = tokio::join!(
            request(&app, "/v1/embeddings", Some(embedding_request.clone())),
            request(&app, "/rerank", Some(rerank_request.clone()))
        );
        assert_eq!(actual_a.0, StatusCode::OK, "{}", actual_a.1);
        assert_eq!(actual_b.0, StatusCode::OK, "{}", actual_b.1);
        assert_eq!(
            actual_a.1,
            request(&single_a, "/v1/embeddings", Some(embedding_request))
                .await
                .1
        );
        assert_eq!(
            actual_b.1,
            request(&single_b, "/rerank", Some(rerank_request.clone()))
                .await
                .1
        );
        assert_eq!(
            request(&app, "/invocations", Some(rerank_request)).await.1,
            actual_b.1
        );
        assert_eq!(
            request(
                &app,
                "/",
                Some(json!({"model":"embed", "inputs":"hello world"}))
            )
            .await
            .0,
            StatusCode::OK
        );
        for payload in [
            json!({"input":"hello"}),
            json!({"model":"missing", "input":"hello"}),
        ] {
            assert_eq!(
                request(&app, "/v1/embeddings", Some(payload)).await.0,
                StatusCode::BAD_REQUEST
            );
        }
        assert_eq!(
            request(
                &app,
                "/v1/systemone",
                Some(json!({"model":"embed", "state":"hello", "questions":{}}))
            )
            .await
            .0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            request(
                &app,
                "/v1/embeddings",
                Some(json!({"model":"rerank", "input":"hello"}))
            )
            .await
            .0,
            StatusCode::BAD_REQUEST
        );
        for name in ["embed", "rerank"] {
            assert_eq!(
                request(
                    &app,
                    "/tokenize",
                    Some(json!({"model":name, "inputs":"hello world"}))
                )
                .await
                .0,
                StatusCode::OK
            );
            assert_eq!(
                request(&app, "/decode", Some(json!({"model":name, "ids":[2,5,3]})))
                    .await
                    .0,
                StatusCode::OK
            );
        }
        let permits: Vec<_> = (0..4)
            .map(|_| embedding.infer.try_acquire_permit().unwrap())
            .collect();
        assert_eq!(
            request(
                &app,
                "/v1/embeddings",
                Some(json!({"model":"embed", "input":"hello"}))
            )
            .await
            .0,
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            request(
                &app,
                "/rerank",
                Some(json!({"model":"rerank", "query":"query", "texts":["document"]}))
            )
            .await
            .0,
            StatusCode::OK
        );
        drop(permits);
        let metrics = handle.render();
        assert!(metrics.contains("model=\"embed\""));
        assert!(metrics.contains("model=\"rerank\""));
        let unauth = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await?;
        assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);
        // Every backend failure participates in aggregate health; one failed model
        // must not prevent a different healthy model from completing requests.
        reranker.shutdown().await;
        assert_eq!(
            request(&app, "/health", None).await.0,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            request(
                &app,
                "/rerank",
                Some(json!({"model":"rerank", "query":"query", "texts":["document"]}))
            )
            .await
            .0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            request(
                &app,
                "/v1/embeddings",
                Some(json!({"model":"embed", "input":"hello"}))
            )
            .await
            .0,
            StatusCode::OK
        );
        tokio::time::timeout(Duration::from_secs(10), registry.shutdown()).await?;
        assert!(!embedding.infer.is_available());
        Ok(())
    }
}
