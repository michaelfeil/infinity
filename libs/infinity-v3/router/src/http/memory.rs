//! Admission before parsing, with reservations retained by response bytes.
use crate::{ErrorResponse, ErrorType};
use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
};
use http_body::{Body as HttpBody, Frame, SizeHint};
use std::{
    io,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone)]
pub(crate) struct MemoryAdmission {
    requests: Arc<Semaphore>,
    inputs: Arc<Semaphore>,
    pub(crate) outputs: Arc<Semaphore>,
    payload_limit: u32,
}
impl MemoryAdmission {
    pub(crate) fn new(
        count: usize,
        payload_limit: usize,
        input_limit: usize,
        output_limit: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            count > 0 && count <= Semaphore::MAX_PERMITS,
            "HTTP concurrency must be positive and representable"
        );
        let payload = u32::try_from(payload_limit)?;
        anyhow::ensure!(
            (64 * 1024..=u32::MAX as usize).contains(&output_limit)
                && output_limit <= Semaphore::MAX_PERMITS,
            "INFINITY_HTTP_OUTPUT_MAX_BYTES must be between 65536 and {} bytes",
            (u32::MAX as usize).min(Semaphore::MAX_PERMITS)
        );
        anyhow::ensure!(
            input_limit >= payload_limit
                && input_limit <= Semaphore::MAX_PERMITS
                && output_limit > 0
                && output_limit <= Semaphore::MAX_PERMITS,
            "HTTP memory budget is unrepresentable or cannot admit one request"
        );
        Ok(Self {
            requests: Arc::new(Semaphore::new(count)),
            inputs: Arc::new(Semaphore::new(input_limit)),
            outputs: Arc::new(Semaphore::new(output_limit)),
            payload_limit: payload,
        })
    }
}
struct AdmissionLease {
    _request: OwnedSemaphorePermit,
    _input: OwnedSemaphorePermit,
}
#[derive(Clone, Copy)]
pub(crate) enum ErrorShape {
    Standard,
    Classify,
    OpenAI,
}
impl ErrorShape {
    pub(crate) fn for_path(path: &str) -> Self {
        match path {
            "/v1/classify" => Self::Classify,
            "/embeddings" | "/v1/embeddings" => Self::OpenAI,
            _ => Self::Standard,
        }
    }
}
tokio::task_local! { static OUTPUTS: Arc<Semaphore>; static SHAPE: ErrorShape; }

fn overloaded(path: &str) -> Response {
    SHAPE.sync_scope(ErrorShape::for_path(path), || {
        let error = ErrorResponse {
            error: "HTTP memory admission capacity exhausted".into(),
            error_type: ErrorType::Overloaded,
        };
        match ErrorShape::for_path(path) {
            ErrorShape::Classify => super::server::classify_inference_error(error).into_response(),
            ErrorShape::OpenAI => <(
                axum::http::StatusCode,
                super::response_format::Json<super::types::OpenAICompatErrorResponse>,
            )>::from(error)
            .into_response(),
            ErrorShape::Standard => <(
                axum::http::StatusCode,
                super::response_format::Json<ErrorResponse>,
            )>::from(error)
            .into_response(),
        }
    })
}
pub(crate) async fn admit(
    State(state): State<MemoryAdmission>,
    request: Request,
    next: Next,
) -> Response {
    // GET model info and infrastructure probes do not parse inference bodies.
    if request.method() != axum::http::Method::POST {
        return next.run(request).await;
    }
    let path = request.uri().path();
    let Ok(slot) = state.requests.clone().try_acquire_owned() else {
        return overloaded(path);
    };
    // Reserve the extractor's entire bounded input before its first body poll.
    let Ok(input) = state
        .inputs
        .clone()
        .try_acquire_many_owned(state.payload_limit)
    else {
        return overloaded(path);
    };
    let lease = Arc::new(AdmissionLease {
        _request: slot,
        _input: input,
    });
    let shape = ErrorShape::for_path(path);
    let mut response = infinity_v3_core::input_memory::with_input_memory(
        lease.clone(),
        SHAPE.scope(shape, next.run(request)),
    )
    .await;
    response.extensions_mut().insert(lease);
    response
}
struct AdmittedBody {
    body: Pin<Box<Body>>,
    lease: Arc<AdmissionLease>,
}
struct AdmittedBytes {
    bytes: Bytes,
    _lease: Arc<AdmissionLease>,
}
impl AsRef<[u8]> for AdmittedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
impl HttpBody for AdmittedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let lease = self.lease.clone();
        self.body.as_mut().poll_frame(cx).map(|frame| {
            frame.map(|r| {
                r.map(|f| {
                    f.map_data(|bytes| {
                        Bytes::from_owner(AdmittedBytes {
                            bytes,
                            _lease: lease,
                        })
                    })
                })
            })
        })
    }
    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

pub(crate) struct ReservedWriter {
    bytes: Vec<u8>,
    permits: Vec<OwnedSemaphorePermit>,
    budget: Option<Arc<Semaphore>>,
    pub(crate) exhausted: bool,
}
impl ReservedWriter {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            permits: Vec::new(),
            budget: OUTPUTS.try_with(Arc::clone).ok(),
            exhausted: false,
        }
    }
    pub(crate) fn finish(self) -> Bytes {
        Bytes::from_owner(ReservedBytes {
            bytes: self.bytes,
            _permits: self.permits,
        })
    }
}
struct ReservedBytes {
    bytes: Vec<u8>,
    _permits: Vec<OwnedSemaphorePermit>,
}
impl AsRef<[u8]> for ReservedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
impl io::Write for ReservedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let required = self.bytes.len().checked_add(bytes.len()).ok_or_else(|| {
            self.exhausted = true;
            io::Error::from(io::ErrorKind::OutOfMemory)
        })?;
        if required > self.bytes.capacity() {
            // Charge allocation capacity, including spare capacity, before allocation.
            let growth = required - self.bytes.capacity();
            let reserve = growth.max(self.bytes.capacity()).max(64 * 1024);
            let target = self.bytes.capacity().checked_add(reserve).ok_or_else(|| {
                self.exhausted = true;
                io::Error::from(io::ErrorKind::OutOfMemory)
            })?;
            // A realloc may keep the old allocation alive while allocating the new
            // buffer. Charge the full new capacity before touching the old buffer.
            let new_permit = if let Some(budget) = &self.budget {
                let count = u32::try_from(target).map_err(|_| {
                    // Geometric growth can exceed the supported budget even when
                    // the configured budget itself fits the permit count.
                    self.exhausted = true;
                    io::Error::from(io::ErrorKind::OutOfMemory)
                })?;
                match budget.clone().try_acquire_many_owned(count) {
                    Ok(permit) => Some(permit),
                    Err(_) => {
                        self.exhausted = true;
                        return Err(io::Error::from(io::ErrorKind::OutOfMemory));
                    }
                }
            } else {
                None
            };
            self.bytes
                .try_reserve_exact(target - self.bytes.len())
                .map_err(|_| {
                    // Allocation pressure uses the shared overload response;
                    // avoid formatting another error while memory is exhausted.
                    self.exhausted = true;
                    io::Error::from(io::ErrorKind::OutOfMemory)
                })?;
            self.permits.clear();
            if let Some(permit) = new_permit {
                self.permits.push(permit);
            }
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(crate) fn error_shape() -> ErrorShape {
    SHAPE
        .try_with(|value| *value)
        .unwrap_or(ErrorShape::Standard)
}
pub(crate) async fn response_budget(
    State(state): State<MemoryAdmission>,
    request: Request,
    next: Next,
) -> Response {
    let response = OUTPUTS.scope(state.outputs, next.run(request)).await;
    let (parts, body) = response.into_parts();
    if let Some(lease) = parts.extensions.get::<Arc<AdmissionLease>>().cloned() {
        Response::from_parts(
            parts,
            Body::new(AdmittedBody {
                body: Box::pin(body),
                lease,
            }),
        )
    } else {
        Response::from_parts(parts, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::post, Router};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;

    async fn large_response(
        super::super::response_format::Json(_): super::super::response_format::Json<
            serde_json::Value,
        >,
    ) -> super::super::response_format::Json<String> {
        super::super::response_format::Json("x".repeat(80_000))
    }
    fn app(state: MemoryAdmission) -> Router {
        Router::new()
            .route("/embed", post(large_response))
            .route("/v1/classify", post(large_response))
            .route("/v1/embeddings", post(large_response))
            .layer(axum::middleware::from_fn_with_state(state.clone(), admit))
            .layer(axum::extract::DefaultBodyLimit::max(1024))
            .layer(axum::middleware::from_fn(
                super::super::response_format::negotiate,
            ))
            .layer(axum::middleware::from_fn_with_state(state, response_budget))
    }
    struct SlowBody {
        polls: Arc<AtomicUsize>,
        polled: Arc<tokio::sync::Notify>,
    }
    impl HttpBody for SlowBody {
        type Data = Bytes;
        type Error = std::convert::Infallible;
        fn poll_frame(
            self: Pin<&mut Self>,
            _: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            self.polled.notify_one();
            Poll::Pending
        }
    }
    fn request(body: Body) -> Request {
        Request::builder()
            .method("POST")
            .uri("/embed")
            .header("content-type", "application/json")
            .body(body)
            .unwrap()
    }
    #[tokio::test]
    async fn slow_upload_is_admitted_before_poll_and_cancel_releases_capacity() {
        for output_limit in [1, 65535] {
            assert!(MemoryAdmission::new(2, 1024, 1024, output_limit).is_err());
        }
        assert!(MemoryAdmission::new(2, 1024, 1024, 65536).is_ok());
        let maximum = (u32::MAX as usize).min(Semaphore::MAX_PERMITS);
        assert!(MemoryAdmission::new(2, 1024, 1024, maximum).is_ok());
        assert!(MemoryAdmission::new(2, 1024, 1024, maximum + 1).is_err());
        let state = MemoryAdmission::new(2, 1024, 1024, 256 * 1024).unwrap();
        let router = app(state.clone());
        let polls = Arc::new(AtomicUsize::new(0));
        let polled = Arc::new(tokio::sync::Notify::new());
        let first = tokio::spawn(router.clone().oneshot(request(Body::new(SlowBody {
            polls: polls.clone(),
            polled: polled.clone(),
        }))));
        tokio::time::timeout(std::time::Duration::from_secs(2), polled.notified())
            .await
            .unwrap();
        assert_eq!(state.requests.available_permits(), 1);
        assert_eq!(state.inputs.available_permits(), 0);
        for path in ["/embed", "/v1/classify", "/v1/embeddings"] {
            let rejected_polls = Arc::new(AtomicUsize::new(0));
            let mut second = request(Body::new(SlowBody {
                polls: rejected_polls.clone(),
                polled: polled.clone(),
            }));
            *second.uri_mut() = path.parse().unwrap();
            let response = router.clone().oneshot(second).await.unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(rejected_polls.load(Ordering::SeqCst), 0);
            let bytes = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            match path {
                "/v1/classify" => {
                    assert_eq!(error["object"], "error");
                    assert_eq!(error["type"], "overloaded_error");
                    assert_eq!(error["code"], 429);
                }
                "/v1/embeddings" => {
                    assert_eq!(error["type"], "Overloaded");
                    assert_eq!(error["code"], 429);
                }
                _ => assert_eq!(error["error_type"], "Overloaded"),
            }
        }
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        assert_eq!(state.requests.available_permits(), 2);
        assert_eq!(state.inputs.available_permits(), 1024);
        let oversize = router
            .oneshot(request(Body::from(format!("\"{}\"", "x".repeat(2048)))))
            .await
            .unwrap();
        assert_eq!(oversize.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
        drop(oversize);
        assert_eq!(state.requests.available_permits(), 2);
        assert_eq!(state.inputs.available_permits(), 1024);
    }
    #[tokio::test]
    async fn response_bytes_keep_count_and_byte_capacity_until_last_transport_clone_drops() {
        let state = MemoryAdmission::new(2, 1024, 2048, 256 * 1024).unwrap();
        let router = app(state.clone());
        let response = router
            .clone()
            .oneshot(request(Body::from("{}")))
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let mut body = Box::pin(response.into_body());
        let frame = std::future::poll_fn(|cx| body.as_mut().poll_frame(cx))
            .await
            .unwrap()
            .unwrap();
        let bytes = frame.into_data().unwrap();
        let transport_clone = bytes.clone();
        drop(body);
        drop(bytes);
        assert_eq!(state.requests.available_permits(), 1);
        assert_eq!(state.outputs.available_permits(), 128 * 1024);
        OUTPUTS
            .scope(state.outputs.clone(), async {
                use std::io::Write;
                let mut growing = ReservedWriter::new();
                growing.write_all(&vec![0; 64 * 1024]).unwrap();
                assert_eq!(state.outputs.available_permits(), 64 * 1024);
                assert!(growing.write_all(&[0]).is_err());
                assert!(growing.exhausted);
                assert_eq!(growing.bytes.capacity(), 64 * 1024);
                assert_eq!(state.outputs.available_permits(), 64 * 1024);
            })
            .await;
        assert_eq!(state.outputs.available_permits(), 128 * 1024);
        let mut second = request(Body::from("{}"));
        second.headers_mut().insert(
            axum::http::header::ACCEPT,
            axum::http::HeaderValue::from_static("application/msgpack"),
        );
        let rejected = router.clone().oneshot(second).await.unwrap();
        assert_eq!(rejected.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            rejected.headers()[axum::http::header::CONTENT_TYPE],
            "application/msgpack"
        );
        let error_bytes = axum::body::to_bytes(rejected.into_body(), 1024)
            .await
            .unwrap();
        let error: serde_json::Value = rmp_serde::from_slice(&error_bytes).unwrap();
        assert_eq!(error["error_type"], "Overloaded");
        drop(error_bytes);
        // Exercise both the input-conversion writer and final-format writer
        // exhausting capacity while a genuine encoded response is still held.
        for held_capacity in [128 * 1024, 64 * 1024] {
            let other_output = state
                .outputs
                .clone()
                .try_acquire_many_owned(held_capacity)
                .unwrap();
            let mut invalid = request(Body::from("{"));
            *invalid.uri_mut() = "/v1/classify".parse().unwrap();
            invalid.headers_mut().insert(
                axum::http::header::ACCEPT,
                axum::http::HeaderValue::from_static("application/msgpack"),
            );
            let converted = router.clone().oneshot(invalid).await.unwrap();
            assert_eq!(
                converted.status(),
                axum::http::StatusCode::TOO_MANY_REQUESTS
            );
            assert_eq!(
                converted.headers()[axum::http::header::CONTENT_TYPE],
                "application/msgpack"
            );
            assert_eq!(state.requests.available_permits(), 0);
            let bytes = axum::body::to_bytes(converted.into_body(), 1024)
                .await
                .unwrap();
            let error: serde_json::Value = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(error["object"], "error");
            assert_eq!(error["type"], "overloaded_error");
            assert_eq!(error["code"], 429);
            assert_eq!(state.requests.available_permits(), 0);
            drop(bytes);
            assert_eq!(state.requests.available_permits(), 1);
            drop(other_output);
        }
        drop(transport_clone);
        assert_eq!(state.requests.available_permits(), 2);
        assert_eq!(state.outputs.available_permits(), 256 * 1024);
        assert_eq!(
            router
                .oneshot(request(Body::from("{}")))
                .await
                .unwrap()
                .status(),
            axum::http::StatusCode::OK
        );
    }
}
