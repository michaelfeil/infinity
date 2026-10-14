//! Typed response serialization and Accept negotiation shared by every HTTP route.
#[cfg(test)]
use axum::body::to_bytes;
use axum::{
    body::Body,
    extract::{rejection::JsonRejection, FromRequest, Request},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::{de::DeserializeOwned, Serialize};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Format {
    Native,
    Json,
    MessagePack,
    LegacyMessagePack,
}

impl Format {
    fn media_type(self) -> &'static str {
        match self {
            Self::Native => unreachable!("native response already has a media type"),
            Self::Json => "application/json",
            Self::MessagePack => "application/msgpack",
            Self::LegacyMessagePack => "application/x-msgpack",
        }
    }
}

tokio::task_local! { static FORMAT: Format; }

/// JSON request extractor with directly serialized JSON or named-map MessagePack responses.
/// Request parsing and rejection types remain Axum's, and OpenAPI retains the original T.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Json<T>(pub T);

#[axum::async_trait]
impl<T: DeserializeOwned, S: Send + Sync> FromRequest<S> for Json<T> {
    type Rejection = JsonRejection;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        axum::Json::<T>::from_request(request, state)
            .await
            .map(|axum::Json(value)| Self(value))
    }
}

impl<T: Serialize> IntoResponse for Json<T> {
    fn into_response(self) -> Response {
        let format = FORMAT.try_with(|format| *format).unwrap_or(Format::Json);
        let mut writer = super::memory::ReservedWriter::new();
        let result = match format {
            Format::Native | Format::Json => {
                serde_json::to_writer(&mut writer, &self.0).map_err(|error| {
                    if writer.exhausted {
                        String::new()
                    } else {
                        error.to_string()
                    }
                })
            }
            Format::MessagePack | Format::LegacyMessagePack => self
                .0
                .serialize(&mut rmp_serde::Serializer::new(&mut writer).with_struct_map())
                .map_err(|error| {
                    if writer.exhausted {
                        String::new()
                    } else {
                        error.to_string()
                    }
                }),
        };
        match result {
            Ok(()) => (
                [(
                    header::CONTENT_TYPE,
                    if format == Format::Native {
                        "application/json"
                    } else {
                        format.media_type()
                    },
                )],
                writer.finish(),
            )
                .into_response(),
            Err(_) if writer.exhausted => {
                // Fixed, small error serialization must not recurse into an exhausted budget.
                memory_error(format, super::memory::error_shape())
            }
            Err(error) => {
                tracing::error!("Response serialization failed: {error}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Response serialization failed",
                )
                    .into_response()
            }
        }
    }
}

// Shared immutable errors keep exhausted-pool responses from allocating one
// serialization buffer per rejected request. Initialized before serving.
static MEMORY_ERRORS: std::sync::OnceLock<[axum::body::Bytes; 3]> = std::sync::OnceLock::new();
const MEMORY_JSON: [&[u8]; 3] = [
    br#"{"error":"HTTP response memory capacity exhausted","error_type":"Overloaded"}"#,
    br#"{"object":"error","message":"HTTP response memory capacity exhausted","type":"overloaded_error","param":null,"code":429}"#,
    br#"{"message":"HTTP response memory capacity exhausted","type":"Overloaded","code":429}"#,
];
pub(crate) fn initialize_memory_errors() {
    MEMORY_ERRORS.get_or_init(|| {
        MEMORY_JSON.map(|bytes| {
            let value: serde_json::Value =
                serde_json::from_slice(bytes).expect("Static memory-error JSON");
            axum::body::Bytes::from(
                rmp_serde::to_vec_named(&value).expect("Static memory-error MessagePack"),
            )
        })
    });
}
fn memory_error(format: Format, shape: super::memory::ErrorShape) -> Response {
    let index = match shape {
        super::memory::ErrorShape::Standard => 0,
        super::memory::ErrorShape::Classify => 1,
        super::memory::ErrorShape::OpenAI => 2,
    };
    let bytes = if matches!(format, Format::MessagePack | Format::LegacyMessagePack) {
        initialize_memory_errors();
        MEMORY_ERRORS.get().unwrap()[index].clone()
    } else {
        axum::body::Bytes::from_static(MEMORY_JSON[index])
    };
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(
            header::CONTENT_TYPE,
            if format == Format::Native {
                "application/json"
            } else {
                format.media_type()
            },
        )],
        bytes,
    )
        .into_response()
}

enum ConversionError {
    Failed,
    Exhausted,
}
fn conversion_error(writer: &super::memory::ReservedWriter) -> ConversionError {
    if writer.exhausted {
        ConversionError::Exhausted
    } else {
        ConversionError::Failed
    }
}

// More specific ranges override wildcards, including explicit q=0 exclusions.
fn preference(headers: &HeaderMap, media_type: &str) -> Option<(u16, u8)> {
    let mut matched = None;
    for value in headers.get_all(header::ACCEPT) {
        let Ok(value) = value.to_str() else { continue };
        for range in value.split(',') {
            let mut parts = range.split(';');
            let range = parts.next().unwrap_or_default().trim();
            let specificity = if range.eq_ignore_ascii_case(media_type) {
                3
            } else if range == "*/*" {
                1
            } else if range.ends_with("/*")
                && media_type
                    .split('/')
                    .next()
                    .is_some_and(|kind| range[..range.len() - 2].eq_ignore_ascii_case(kind))
            {
                2
            } else {
                continue;
            };
            let mut quality = 1000;
            for parameter in parts {
                if let Some((name, value)) = parameter.trim().split_once('=') {
                    if name.trim().eq_ignore_ascii_case("q") {
                        quality = value
                            .trim()
                            .parse::<f32>()
                            .ok()
                            .filter(|q| q.is_finite() && (0.0..=1.0).contains(q))
                            .map_or(0, |q| (q * 1000.0).round() as u16);
                    }
                }
            }
            matched = Some(matched.unwrap_or((0, 0)).max((specificity, quality)));
        }
    }
    matched.map(|(specificity, quality)| (quality, specificity))
}

fn select(headers: &HeaderMap, native: Option<&str>) -> Option<Format> {
    if !headers.contains_key(header::ACCEPT) {
        return Some(Format::Native);
    }
    // Prefer the native representation on ties. Binary conversion requires a
    // higher quality or a more specific match than the native/JSON fallback.
    let mut selected = None;
    let mut best = (0, 0);
    for (format, media_type) in [
        (Format::Native, native.unwrap_or("application/json")),
        (Format::Json, "application/json"),
        (Format::MessagePack, "application/msgpack"),
        (Format::LegacyMessagePack, "application/x-msgpack"),
    ] {
        if let Some(rank) = preference(headers, media_type) {
            if rank.0 > 0 && rank > best {
                best = rank;
                selected = Some(format);
            }
        }
    }
    selected
}

pub(crate) async fn negotiate(request: Request, next: Next) -> Response {
    let memory_error_shape = super::memory::ErrorShape::for_path(request.uri().path());
    let headers = request.headers().clone();
    let initial = select(&headers, None);
    // Native text routes can satisfy Accept values outside the API formats.
    let native_route =
        request.uri().path().starts_with("/docs") || request.uri().path() == "/metrics";
    if initial.is_none() && !native_route {
        let mut response = StatusCode::NOT_ACCEPTABLE.into_response();
        response
            .headers_mut()
            .append(header::VARY, HeaderValue::from_static("Accept"));
        return response;
    }
    let head = request.method() == Method::HEAD;
    let response = FORMAT
        .scope(initial.unwrap_or(Format::Native), next.run(request))
        .await;
    let (mut parts, body) = response.into_parts();
    parts
        .headers
        .append(header::VARY, HeaderValue::from_static("Accept"));
    if parts.status == StatusCode::NO_CONTENT
        || parts.status == StatusCode::NOT_MODIFIED
        || parts.status.is_informational()
    {
        return Response::from_parts(parts, body);
    }
    let content_type = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok());
    let native = content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim);
    let Some(format) = select(&headers, native) else {
        parts.status = StatusCode::NOT_ACCEPTABLE;
        parts.headers.remove(header::CONTENT_TYPE);
        parts.headers.remove(header::CONTENT_LENGTH);
        return Response::from_parts(parts, Body::empty());
    };
    if format == Format::Native || native == Some(format.media_type()) {
        return Response::from_parts(parts, body);
    }
    let original_json = native == Some("application/json");
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(format.media_type()),
    );
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.remove(header::ETAG);
    if head {
        return Response::from_parts(parts, Body::empty());
    }
    // Only untyped text/empty/framework responses need buffering. Typed API responses
    // are serialized directly above, preserving f32 and integer types without JSON parsing.
    let encoded = async {
        // Untyped framework/document responses use a finite conversion limit.
        // Charge buffered input before allocation; typed inference bodies bypass this path.
        use http_body::Body as _;
        use std::io::Write;
        let mut body = std::pin::pin!(body);
        let mut writer = super::memory::ReservedWriter::new();
        let mut length = 0usize;
        while let Some(frame) = std::future::poll_fn(|cx| body.as_mut().poll_frame(cx)).await {
            let frame = frame.map_err(|_| ConversionError::Failed)?;
            if let Ok(data) = frame.into_data() {
                length = length
                    .checked_add(data.len())
                    .ok_or(ConversionError::Failed)?;
                if length > 2 * 1024 * 1024 {
                    return Err(ConversionError::Failed);
                }
                writer
                    .write_all(&data)
                    .map_err(|_| conversion_error(&writer))?;
            }
        }
        let bytes = writer.finish();
        let value = if original_json {
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .map_err(|_| ConversionError::Failed)?
        } else if bytes.is_empty() {
            serde_json::Value::Null
        } else if let Ok(text) = std::str::from_utf8(&bytes) {
            serde_json::Value::String(text.to_owned())
        } else {
            serde_json::to_value(bytes.to_vec()).map_err(|_| ConversionError::Failed)?
        };
        let mut encoded = super::memory::ReservedWriter::new();
        if format == Format::Json {
            serde_json::to_writer(&mut encoded, &value).map_err(|_| conversion_error(&encoded))?;
        } else {
            value
                .serialize(&mut rmp_serde::Serializer::new(&mut encoded).with_struct_map())
                .map_err(|_| conversion_error(&encoded))?;
        }
        Ok::<_, ConversionError>(encoded.finish())
    }
    .await;
    match encoded {
        Ok(bytes) => Response::from_parts(parts, Body::from(bytes)),
        Err(ConversionError::Exhausted) => {
            let (error_parts, error_body) = memory_error(format, memory_error_shape).into_parts();
            parts.status = error_parts.status;
            parts.headers.extend(error_parts.headers);
            // Retain the admission lease in the original response extensions.
            Response::from_parts(parts, error_body)
        }
        Err(ConversionError::Failed) => {
            parts.status = StatusCode::INTERNAL_SERVER_ERROR;
            Response::from_parts(parts, Body::empty())
        }
    }
}

pub(crate) fn document(doc: &mut utoipa::openapi::OpenApi) {
    for (name, path) in &mut doc.paths.paths {
        for operation in path.operations.values_mut() {
            for response in operation.responses.responses.values_mut() {
                let utoipa::openapi::RefOr::T(response) = response else {
                    continue;
                };
                // Empty success responses are still bodyless by default, but encode
                // null when a JSON or MessagePack representation is requested.
                if name == "/health" && response.content.is_empty() {
                    let schema = utoipa::openapi::schema::ObjectBuilder::new()
                        .schema_type(utoipa::openapi::schema::SchemaType::Value)
                        .nullable(true)
                        .enum_values(Some([serde_json::Value::Null]))
                        .build();
                    response.content.insert(
                        "application/json".into(),
                        utoipa::openapi::ContentBuilder::new()
                            .schema(schema)
                            .build(),
                    );
                }
                if !response.content.contains_key("application/json") {
                    if let Some(content) = response.content.get("text/plain").cloned() {
                        response.content.insert("application/json".into(), content);
                    }
                }
                if let Some(content) = response
                    .content
                    .get("application/json")
                    .or_else(|| response.content.get("text/plain"))
                {
                    let content = utoipa::openapi::ContentBuilder::new()
                        .schema(content.schema.clone())
                        .build();
                    response
                        .content
                        .insert("application/msgpack".into(), content);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        routing::{get, post},
        Router,
    };
    use serde::Deserialize;
    use tower::ServiceExt;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Payload {
        integer: u64,
        score: f32,
    }

    fn app() -> Router {
        Router::new()
            .route(
                "/info",
                get(|| async {
                    tokio::task::yield_now().await;
                    Json(Payload {
                        integer: u64::MAX,
                        score: 0.5,
                    })
                }),
            )
            .route(
                "/echo",
                post(|Json(value): Json<Payload>| async move { Json(value) }),
            )
            .route(
                "/error",
                get(|| async {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({"error":"bad input"})),
                    )
                }),
            )
            .route("/metrics", get(|| async { "requests 1\n" }))
            .route(
                "/docs",
                get(|| async { axum::response::Html("<html>docs</html>") }),
            )
            .route("/health", get(|| async {}))
            .route("/empty", get(|| async { StatusCode::NO_CONTENT }))
            .layer(axum::middleware::from_fn(negotiate))
    }

    async fn call(path: &str, accept: Option<&str>) -> Response {
        let mut request = Request::builder().uri(path);
        if let Some(accept) = accept {
            request = request.header(header::ACCEPT, accept);
        }
        app()
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    #[test]
    fn accept_preferences_preserve_native_defaults() {
        for (accept, expected) in [
            ("*/*", Some(Format::Native)),
            ("application/*", Some(Format::Native)),
            (
                "application/json,application/msgpack;q=0.1",
                Some(Format::Native),
            ),
            ("APPLICATION/X-MSGPACK", Some(Format::LegacyMessagePack)),
            (
                "application/msgpack;q=0,application/x-msgpack;q=0,application/*",
                Some(Format::Native),
            ),
            ("application/xml", None),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(header::ACCEPT, accept.parse().unwrap());
            assert_eq!(select(&headers, None), expected, "{accept}");
        }
    }

    #[tokio::test]
    async fn browser_and_generic_accept_headers_preserve_native_bodies() {
        for accept in ["*/*", "text/html,*/*;q=0.1"] {
            let response = call("/docs", Some(accept)).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert!(response.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/html"));
            assert_eq!(
                to_bytes(response.into_body(), 1024).await.unwrap(),
                "<html>docs</html>"
            );
        }
        for accept in [
            "*/*",
            "application/*",
            "application/json,application/msgpack;q=0.1",
        ] {
            let response = call("/info", Some(accept)).await;
            assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
            serde_json::from_slice::<Payload>(&to_bytes(response.into_body(), 1024).await.unwrap())
                .unwrap();
        }
        let response = call("/info", Some("application/json;q=0,*/*;q=1")).await;
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/msgpack"
        );
        let response = call(
            "/info",
            Some("application/msgpack;q=0,application/x-msgpack;q=0,application/json;q=0,*/*"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_ACCEPTABLE);
    }

    #[tokio::test]
    async fn typed_responses_preserve_numeric_types_and_request_isolation() {
        let (json, msgpack, alias) = tokio::join!(
            call("/info", None),
            call("/info", Some("application/json;q=0.1,application/msgpack")),
            call("/info", Some("application/x-msgpack"))
        );
        assert_eq!(json.headers()[header::CONTENT_TYPE], "application/json");
        assert_eq!(
            alias.headers()[header::CONTENT_TYPE],
            "application/x-msgpack"
        );
        assert_eq!(
            msgpack.headers()[header::CONTENT_TYPE],
            "application/msgpack"
        );
        assert_eq!(msgpack.headers()[header::VARY], "Accept");
        let bytes = to_bytes(msgpack.into_body(), 1024).await.unwrap();
        assert_eq!(bytes[0], 0x82); // Named map, not a positional struct array.
        assert!(bytes.ends_with(&[0xca, 0x3f, 0, 0, 0])); // Native f32, without JSON/f64 conversion.
        let decoded = rmp_serde::from_slice::<Payload>(&bytes).unwrap();
        assert_eq!(
            decoded,
            Payload {
                integer: u64::MAX,
                score: 0.5
            }
        );
        assert_eq!(
            serde_json::from_slice::<Payload>(&to_bytes(json.into_body(), 1024).await.unwrap())
                .unwrap(),
            decoded
        );
    }

    #[tokio::test]
    async fn errors_text_and_bodyless_responses() {
        let error = call("/error", Some("application/msgpack")).await;
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            rmp_serde::from_slice::<serde_json::Value>(
                &to_bytes(error.into_body(), 1024).await.unwrap()
            )
            .unwrap()["error"],
            "bad input"
        );
        let malformed = Request::builder()
            .method(Method::POST)
            .uri("/echo")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, "application/msgpack")
            .body(Body::from("bad JSON"))
            .unwrap();
        let error = app().oneshot(malformed).await.unwrap();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert!(
            rmp_serde::from_slice::<String>(&to_bytes(error.into_body(), 1024).await.unwrap())
                .unwrap()
                .contains("Failed to parse")
        );

        let native = call("/metrics", None).await;
        assert_eq!(
            to_bytes(native.into_body(), 1024).await.unwrap(),
            "requests 1\n"
        );
        let encoded = call("/metrics", Some("*/*")).await;
        assert_eq!(
            to_bytes(encoded.into_body(), 1024).await.unwrap(),
            "requests 1\n"
        );
        assert!(
            call("/docs", Some("text/html")).await.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        assert_eq!(
            call("/docs", Some("text/html,*/*;q=0.1")).await.headers()[header::CONTENT_TYPE],
            "text/html; charset=utf-8"
        );
        let health = call("/health", Some("application/msgpack")).await;
        assert_eq!(
            to_bytes(health.into_body(), 1024).await.unwrap().as_ref(),
            &[0xc0]
        );
        let empty = call("/empty", Some("application/msgpack")).await;
        assert_eq!(empty.status(), StatusCode::NO_CONTENT);
        assert!(to_bytes(empty.into_body(), 1024).await.unwrap().is_empty());
        assert_eq!(
            call("/info", Some("application/xml")).await.status(),
            StatusCode::NOT_ACCEPTABLE
        );
        let head = Request::builder()
            .method(Method::HEAD)
            .uri("/health")
            .header(header::ACCEPT, "application/msgpack")
            .body(Body::empty())
            .unwrap();
        let response = app().oneshot(head).await.unwrap();
        assert!(to_bytes(response.into_body(), 1024)
            .await
            .unwrap()
            .is_empty());
    }
}

/// Preserve native host-result ownership across serialization and body handoff.
pub(crate) async fn retain_results(request: Request, next: Next) -> Response {
    use infinity_v3_backend::result_memory::{RequestResults, REQUEST_RESULTS};
    use std::sync::Arc;
    let owner = Arc::new(RequestResults::default());
    let response = REQUEST_RESULTS
        .scope(owner.clone(), next.run(request))
        .await;
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(ResultBody {
            body: Box::pin(body),
            owner,
        }),
    )
}
struct ResultBody {
    body: std::pin::Pin<Box<Body>>,
    owner: std::sync::Arc<infinity_v3_backend::result_memory::RequestResults>,
}
struct ResultChunk {
    bytes: axum::body::Bytes,
    _owner: std::sync::Arc<infinity_v3_backend::result_memory::RequestResults>,
}
impl AsRef<[u8]> for ResultChunk {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
impl http_body::Body for ResultBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;
    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        this.body.as_mut().poll_frame(cx).map(|frame| {
            frame.map(|result| {
                result.map(|frame| {
                    frame.map_data(|bytes| {
                        axum::body::Bytes::from_owner(ResultChunk {
                            bytes,
                            _owner: this.owner.clone(),
                        })
                    })
                })
            })
        })
    }
    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }
    fn size_hint(&self) -> http_body::SizeHint {
        self.body.size_hint()
    }
}

#[cfg(test)]
mod result_lifetime_tests {
    use super::*;
    use axum::{extract::State, routing::get, Router};
    use infinity_v3_backend::result_memory::{retain_for_request, ResultPool};
    use std::sync::Arc;
    use tower::ServiceExt;

    async fn result(
        State(pool): State<Arc<ResultPool>>,
    ) -> Result<Json<Vec<f32>>, (StatusCode, Json<crate::ErrorResponse>)> {
        let memory = pool.reserve(128).map_err(|e| {
            crate::ErrorResponse::from(infinity_v3_core::TextEmbeddingsError::Backend(e))
        })?;
        let values = vec![1.0f32; 32];
        retain_for_request(Some(memory));
        Ok(Json(values))
    }
    #[tokio::test]
    async fn retained_result_ownership_rejects_then_recovers_after_transport_clone_drop() {
        use futures::StreamExt;
        let pool = ResultPool::new(128).unwrap();
        let app = Router::new()
            .route("/", get(result))
            .with_state(pool.clone())
            .layer(axum::middleware::from_fn(retain_results));
        let request = || Request::builder().uri("/").body(Body::empty()).unwrap();
        let first = app.clone().oneshot(request()).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let mut stream = first.into_body().into_data_stream();
        let bytes = stream.next().await.unwrap().unwrap();
        let transport_clone = bytes.clone();
        drop(stream);
        drop(bytes);
        assert_eq!(pool.used(), 128);
        let rejected = app.clone().oneshot(request()).await.unwrap();
        assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
        drop(rejected);
        drop(transport_clone);
        assert_eq!(pool.used(), 0);
        let recovered = app.oneshot(request()).await.unwrap();
        assert_eq!(recovered.status(), StatusCode::OK);
        drop(recovered);
        assert_eq!(pool.used(), 0);
    }
}
