//! Parse each request once, then choose its resident model before tokenization.
use super::{response_format::Json, types::OpenAICompatErrorResponse};
use crate::{model_registry::ModelRegistry, ErrorResponse, ErrorType, ModelRuntime};
use axum::{
    extract::{rejection::JsonRejection, FromRequest, Request},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::de::DeserializeOwned;
use std::sync::Arc;

pub(crate) trait RequestedModel {
    fn requested_model(&self) -> Result<Option<&str>, String>;
}

pub(crate) struct RoutedJson<T> {
    pub runtime: Arc<ModelRuntime>,
    pub value: T,
}

pub(crate) enum RoutingRejection {
    Json(JsonRejection),
    Model {
        message: String,
        openai: bool,
        status: StatusCode,
    },
}
impl RoutingRejection {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Json(error) => error.status(),
            Self::Model { status, .. } => *status,
        }
    }
    pub fn body_text(&self) -> String {
        match self {
            Self::Json(error) => error.body_text(),
            Self::Model { message, .. } => message.clone(),
        }
    }
}
impl IntoResponse for RoutingRejection {
    fn into_response(self) -> Response {
        match self {
            Self::Json(error) => error.into_response(),
            Self::Model {
                message,
                openai,
                status,
            } => {
                let error = ErrorResponse {
                    error: message,
                    error_type: if status == StatusCode::SERVICE_UNAVAILABLE {
                        ErrorType::Unhealthy
                    } else {
                        ErrorType::Validation
                    },
                };
                if openai {
                    (
                        status,
                        Json(OpenAICompatErrorResponse {
                            message: error.error,
                            code: status.as_u16(),
                            error_type: error.error_type,
                        }),
                    )
                        .into_response()
                } else {
                    (status, Json(error)).into_response()
                }
            }
        }
    }
}

#[axum::async_trait]
impl<T: DeserializeOwned + RequestedModel + Send, S: Send + Sync> FromRequest<S> for RoutedJson<T> {
    type Rejection = RoutingRejection;
    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let registry = request
            .extensions()
            .get::<ModelRegistry>()
            .expect("model registry installed")
            .clone();
        let path = request.uri().path().to_owned();
        let openai = matches!(path.as_str(), "/embeddings" | "/v1/embeddings");
        let Json(value) = Json::<T>::from_request(request, state)
            .await
            .map_err(RoutingRejection::Json)?;
        let name = value
            .requested_model()
            .map_err(|message| RoutingRejection::Model {
                message,
                openai,
                status: StatusCode::BAD_REQUEST,
            })?;
        let runtime = registry
            .resolve(name)
            .map_err(|message| RoutingRejection::Model {
                message,
                openai,
                status: StatusCode::BAD_REQUEST,
            })?;
        use crate::ModelType;
        let dense_embedding = matches!(&runtime.info.model_type, ModelType::Embedding(model) if model.pooling != "splade");
        let supported = match path.as_str() {
            "/embed" | "/embed_all" | "/similarity" | "/embeddings" | "/v1/embeddings" => {
                dense_embedding
            }
            "/embed_sparse" => {
                matches!(&runtime.info.model_type, ModelType::Embedding(model) if model.pooling == "splade")
            }
            "/predict" | "/predict_tokens" | "/v1/classify" => matches!(
                &runtime.info.model_type,
                ModelType::Classifier(_) | ModelType::Reranker(_)
            ),
            "/rerank" => matches!(&runtime.info.model_type, ModelType::Reranker(_)),
            "/v1/systemone" => matches!(&runtime.info.model_type, ModelType::Decision),
            _ => true,
        };
        if !supported {
            return Err(RoutingRejection::Model {
                message: format!("Model {} does not support {path}", runtime.info.model_id),
                openai,
                status: StatusCode::BAD_REQUEST,
            });
        }
        if !runtime.infer.is_available() {
            return Err(RoutingRejection::Model {
                message: format!("Model {} is unhealthy", runtime.info.model_id),
                openai,
                status: StatusCode::SERVICE_UNAVAILABLE,
            });
        }
        Ok(Self { runtime, value })
    }
}
impl RequestedModel for serde_json::Value {
    fn requested_model(&self) -> Result<Option<&str>, String> {
        match self.get("model") {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::String(name)) => Ok(Some(name)),
            _ => Err("model must be a string".into()),
        }
    }
}
