//! Portable packed embedding tensors. All multi-byte values use little endian.
use super::result_storage;
use super::types::{AllEmbedding, Embedding, EncodingFormat};
use base64::{display::Base64Display, engine::general_purpose::STANDARD, Engine};
use infinity_v3_backend::BackendError;
use infinity_v3_core::TextEmbeddingsError;
use serde::{Serialize, Serializer};
use utoipa::{
    openapi::{
        schema::{KnownFormat, Schema, SchemaFormat},
        RefOr,
    },
    ToSchema,
};

/// Packed public output values, independent of the backend's internal precision.
/// Storage is contiguous C-order, little-endian, without padding or a tensor header.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct Tensor {
    /// Element counts: [dimensions] or [tokens, dimensions]; never byte counts.
    #[schema(min_items = 1, max_items = 2)]
    shape: Vec<usize>,
    /// Public output dtype, not the model's compute or weight dtype. Only float32 is supported.
    #[schema(inline)]
    dtype: TensorDType,
    /// "base64" for JSON, "binary" for MessagePack.
    #[schema(schema_with = data_encoding_schema)]
    data_encoding: DataEncoding,
    /// Exactly 4 * product(shape) decoded bytes: little-endian IEEE 754 float32,
    /// contiguous row-major, with no padding. JSON carries standard padded base64;
    /// MessagePack carries a bin value, never an array of numbers.
    #[schema(value_type = String, format = Byte)]
    data: TensorData,
}

/// Public output dtype, independent of backend precision. Only float32 is supported.
#[derive(Debug, Serialize, ToSchema)]
enum TensorDType {
    #[serde(rename = "float32")]
    Float32,
}

fn data_encoding_schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
    utoipa::openapi::ObjectBuilder::new()
        .schema_type(utoipa::openapi::schema::SchemaType::String)
        .enum_values(Some(["base64"]))
        .into()
}

#[derive(Debug)]
struct DataEncoding;
impl Serialize for DataEncoding {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let encoding = if serializer.is_human_readable() {
            "base64"
        } else {
            "binary"
        };
        serializer.serialize_str(encoding)
    }
}

#[derive(Debug)]
struct TensorData(Vec<u8>);
impl Serialize for TensorData {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            // Stream base64 into the response writer without another full-size string.
            serializer.collect_str(&Base64Display::new(&self.0, &STANDARD))
        } else {
            serializer.serialize_bytes(&self.0)
        }
    }
}

fn invalid(message: &str) -> TextEmbeddingsError {
    BackendError::Inference(message.into()).into()
}

impl Tensor {
    pub(crate) fn vector(values: &[f32]) -> Result<Self, TextEmbeddingsError> {
        Self::new(&[values.len()], std::iter::once(values))
    }

    pub(crate) fn matrix(rows: &[Vec<f32>]) -> Result<Self, TextEmbeddingsError> {
        let width = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|row| row.len() != width) {
            return Err(invalid(
                "Embedding tensor rows have inconsistent dimensions",
            ));
        }
        Self::new(&[rows.len(), width], rows.iter().map(Vec::as_slice))
    }

    fn new<'a>(
        shape: &[usize],
        rows: impl Iterator<Item = &'a [f32]>,
    ) -> Result<Self, TextEmbeddingsError> {
        let elements = shape
            .iter()
            .try_fold(1usize, |size, dimension| size.checked_mul(*dimension))
            .ok_or_else(|| invalid("Embedding tensor size overflow"))?;
        let size = elements
            .checked_mul(4)
            .ok_or_else(|| invalid("Embedding tensor size overflow"))?;
        let mut dimensions = result_storage::vector(shape.len())?;
        dimensions.extend_from_slice(shape);
        let mut data = result_storage::vector(size)?;
        for row in rows {
            for value in row {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(Self {
            shape: dimensions,
            dtype: TensorDType::Float32,
            data_encoding: DataEncoding,
            data: TensorData(data),
        })
    }
}

pub(crate) fn encode_vector(
    values: Vec<f32>,
    format: EncodingFormat,
) -> Result<Embedding, TextEmbeddingsError> {
    match format {
        EncodingFormat::Float => Ok(Embedding::Float(values)),
        EncodingFormat::Tensor => Ok(Embedding::All(AllEmbedding::Tensor(Tensor::vector(
            &values,
        )?))),
        EncodingFormat::Base64 => {
            // Preserve the legacy base64 representation (native-endian float32).
            let bytes = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
            };
            let size = base64::encoded_len(bytes.len(), true)
                .ok_or_else(|| invalid("Base64 output size overflow"))?;
            let mut output = result_storage::string_capacity(size)?;
            STANDARD.encode_string(bytes, &mut output);
            Ok(Embedding::Base64(output))
        }
    }
}

pub(crate) fn encode_matrix(
    values: Vec<Vec<f32>>,
    format: EncodingFormat,
) -> Result<AllEmbedding, TextEmbeddingsError> {
    match format {
        EncodingFormat::Float => Ok(AllEmbedding::Float(values)),
        EncodingFormat::Tensor => Ok(AllEmbedding::Tensor(Tensor::matrix(&values)?)),
        EncodingFormat::Base64 => {
            let mut rows = result_storage::vector(values.len())?;
            for row in values {
                let Embedding::Base64(encoded) = encode_vector(row, format)? else {
                    unreachable!()
                };
                rows.push(encoded);
            }
            Ok(AllEmbedding::Base64(rows))
        }
    }
}

// Documentation only: specialize the tensor response graph without changing serialization.
const TENSOR_SCHEMAS: &[&str] = &[
    "Tensor",
    "AllEmbedding",
    "Embedding",
    "EmbedResponse",
    "EmbedAllResponse",
    "OpenAICompatEmbedding",
    "OpenAICompatResponse",
];

pub(super) fn messagepack_schema(schema: &mut RefOr<Schema>) {
    match schema {
        RefOr::Ref(reference) => {
            if reference
                .ref_location
                .strip_prefix("#/components/schemas/")
                .is_some_and(|name| TENSOR_SCHEMAS.contains(&name))
            {
                reference.ref_location.push_str("MessagePack");
            }
        }
        RefOr::T(Schema::Object(object)) => {
            for property in object.properties.values_mut() {
                messagepack_schema(property);
            }
        }
        RefOr::T(Schema::Array(array)) => messagepack_schema(&mut array.items),
        RefOr::T(composite) => {
            let items = match composite {
                Schema::OneOf(value) => &mut value.items,
                Schema::AllOf(value) => &mut value.items,
                Schema::AnyOf(value) => &mut value.items,
                _ => return,
            };
            for item in items {
                messagepack_schema(item);
            }
        }
    }
}

pub(super) fn document(doc: &mut utoipa::openapi::OpenApi) {
    let Some(components) = doc.components.as_mut() else {
        return;
    };
    for name in TENSOR_SCHEMAS {
        let Some(mut schema) = components.schemas.get(*name).cloned() else {
            continue;
        };
        messagepack_schema(&mut schema);
        if let ("Tensor", RefOr::T(Schema::Object(object))) = (*name, &mut schema) {
            if let Some(RefOr::T(Schema::Object(data))) = object.properties.get_mut("data") {
                data.format = Some(SchemaFormat::KnownFormat(KnownFormat::Binary));
            }
            if let Some(RefOr::T(Schema::Object(encoding))) =
                object.properties.get_mut("data_encoding")
            {
                encoding.enum_values = Some(vec![serde_json::json!("binary")]);
            }
        }
        components
            .schemas
            .insert(format!("{name}MessagePack"), schema);
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{
        EmbedAllRequest, EmbedAllResponse, EmbedRequest, EmbedResponse, OpenAICompatRequest,
    };
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
        routing::get,
        Router,
    };
    use serde::Deserialize;
    use serde_json::json;
    use tower::ServiceExt;

    #[derive(Deserialize)]
    struct WireTensor<'a> {
        shape: Vec<usize>,
        dtype: &'a str,
        data_encoding: &'a str,
        #[serde(borrow)]
        data: &'a [u8], // Borrowing requires a MessagePack bin value.
    }

    #[tokio::test]
    async fn negotiated_tensors_preserve_float_bits_and_little_endian_bytes() {
        let app = Router::new()
            .route(
                "/tensor",
                get(|| async {
                    super::super::response_format::Json(
                        Tensor::vector(&[1., -2., -0., f32::from_bits(0x7fc01234)]).unwrap(),
                    )
                }),
            )
            .layer(axum::middleware::from_fn(
                super::super::response_format::negotiate,
            ));
        let expected = [
            0, 0, 128, 63, 0, 0, 0, 192, 0, 0, 0, 128, 0x34, 0x12, 0xc0, 0x7f,
        ];
        for accept in [
            "application/json",
            "application/msgpack",
            "application/x-msgpack",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/tensor")
                        .header("accept", accept)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.headers()["content-type"], accept);
            let body = to_bytes(response.into_body(), 4096).await.unwrap();
            if accept == "application/json" {
                let tensor: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(tensor["shape"], json!([4]));
                assert_eq!(tensor["dtype"], "float32");
                assert_eq!(tensor["data_encoding"], "base64");
                assert_eq!(
                    STANDARD.decode(tensor["data"].as_str().unwrap()).unwrap(),
                    expected
                );
            } else {
                let tensor: WireTensor = rmp_serde::from_slice(&body).unwrap();
                assert_eq!(tensor.shape, [4]);
                assert_eq!(tensor.dtype, "float32");
                assert_eq!(tensor.data_encoding, "binary");
                assert_eq!(tensor.data, expected);
            }
        }
    }

    #[test]
    fn late_interaction_batches_preserve_shapes_and_token_order() {
        let matrices = [vec![vec![1., 2.], vec![3., 4.]], vec![vec![-1., -2.]]];
        let tensors = matrices
            .into_iter()
            .map(|matrix| encode_matrix(matrix, EncodingFormat::Tensor).unwrap())
            .collect();
        let packed = rmp_serde::to_vec_named(&EmbedAllResponse(tensors)).unwrap();
        let decoded: Vec<WireTensor> = rmp_serde::from_slice(&packed).unwrap();
        assert_eq!(decoded[0].shape, [2, 2]);
        assert_eq!(decoded[1].shape, [1, 2]);
        let values: Vec<f32> = decoded[0]
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, [1., 2., 3., 4.]);
        assert_eq!(decoded[1].data.len(), 8);
        assert!(Tensor::matrix(&[vec![1.], vec![2., 3.]]).is_err());
        assert_eq!(Tensor::matrix(&[]).unwrap().shape, [0, 0]);
        assert!(Tensor::new(&[usize::MAX], std::iter::empty()).is_err());
    }

    #[test]
    fn legacy_formats_and_request_defaults_are_unchanged() {
        let dense = EmbedResponse(vec![
            encode_vector(vec![1., -2.], EncodingFormat::Float).unwrap()
        ]);
        assert_eq!(serde_json::to_value(dense).unwrap(), json!([[1., -2.]]));
        let matrix = EmbedAllResponse(vec![encode_matrix(
            vec![vec![1., -2.]],
            EncodingFormat::Float,
        )
        .unwrap()]);
        assert_eq!(serde_json::to_value(matrix).unwrap(), json!([[[1., -2.]]]));
        let encoded =
            serde_json::to_value(encode_vector(vec![1., -2.], EncodingFormat::Base64).unwrap())
                .unwrap();
        assert_eq!(
            STANDARD.decode(encoded.as_str().unwrap()).unwrap(),
            [1.0f32.to_ne_bytes(), (-2.0f32).to_ne_bytes()].concat()
        );
        let native = json!({"inputs":"hello", "encoding_format":"tensor"});
        let embed: EmbedRequest = serde_json::from_value(native.clone()).unwrap();
        let all: EmbedAllRequest = serde_json::from_value(native).unwrap();
        let openai: OpenAICompatRequest =
            serde_json::from_value(json!({"input":"hello", "encoding_format":"tensor"})).unwrap();
        assert_eq!(
            [
                embed.encoding_format,
                all.encoding_format,
                openai.encoding_format
            ],
            [EncodingFormat::Tensor; 3]
        );
        assert_eq!(
            serde_json::from_value::<EmbedRequest>(json!({"inputs":"hi"}))
                .unwrap()
                .encoding_format,
            EncodingFormat::Float
        );
    }

    #[tokio::test]
    async fn tensor_buffers_respect_the_result_memory_budget() {
        use infinity_v3_backend::result_memory::{RequestResults, ResultPool, REQUEST_RESULTS};
        use std::sync::Arc;
        let pool = ResultPool::new(128).unwrap();
        let owner = Arc::new(RequestResults::default());
        owner.retain(Some(pool.reserve(16).unwrap()));
        REQUEST_RESULTS
            .scope(owner, async {
                assert!(matches!(
                    Tensor::vector(&[0.; 64]),
                    Err(TextEmbeddingsError::Backend(BackendError::Overloaded(_)))
                ));
            })
            .await;
        assert_eq!(pool.used(), 0);
    }
}
