# Embedding formats and client usage

Choose the response format that fits your application: JSON arrays for easy
integration, or packed tensors for NumPy views without a per-element conversion.
Requests remain JSON in every case.

## Choose a format

| `encoding_format` | Best for | Response |
| --- | --- | --- |
| `float` (default) | Standard OpenAI-compatible clients | JSON arrays of numbers |
| `base64` | Clients that decode packed vectors | Base64-encoded vectors |
| `tensor` | NumPy and variable-length token embeddings | A buffer with explicit shape and dtype |

The `tensor` format is available in Infinity v3 on `/embeddings`,
`/v1/embeddings`, `/embed` and `/embed_all`. It does not change pooling,
normalization or truncation. Include `model` when serving multiple models.
The legacy `base64` format retains native-endian float32 for compatibility;
the explicit portable byte-order guarantee applies to `tensor`.

## Python: Baseten Performance Client

For batched embedding workloads, use
[baseten-performance-client](https://github.com/basetenlabs/truss/tree/main/baseten-performance-client).
Its Rust implementation handles concurrent requests and exposes results as NumPy
arrays. It works with a local Infinity server as well as hosted endpoints.

```bash
pip install baseten-performance-client numpy
```

```python
from baseten_performance_client import PerformanceClient, RequestProcessingPreference

client = PerformanceClient(
    base_url="http://localhost:8000",
    api_key="unused",  # For a local server without authentication; otherwise use its API key.
)
response = client.embed(
    input=["A mountain lake.", "An alpine lake."],
    model="sentence-transformers/all-MiniLM-L6-v2",
    encoding_format="float",
    preference=RequestProcessingPreference(batch_size=32, max_concurrent_requests=8),
)
embeddings = response.numpy()  # shape: [inputs, dimensions]
```

Version 0.1.16 assembles the vectors into a contiguous array in Rust; `.numpy()`
is not a zero-copy view of the response bytes. For that, use v3's tensor format
and `np.frombuffer` as below. The client's typed embedding response does not yet
support this tensor envelope.

## Zero-copy NumPy views

Request tensors with MessagePack to receive binary buffers directly, without
base64 decoding. `np.frombuffer` views a buffer as float32 values; `reshape`
restores its dimensions without copying the values.

```bash
pip install requests msgpack numpy
```

```python
import msgpack
import numpy as np
import requests

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers={"Accept": "application/msgpack"},
    json={"input": ["A mountain lake."], "encoding_format": "tensor"},
)
response.raise_for_status()
tensor = msgpack.unpackb(response.content, raw=False)["data"][0]["embedding"]
embedding = np.frombuffer(tensor["data"], dtype="<f4").reshape(tensor["shape"])

print(embedding.shape)          # (dimensions,)
print(embedding.flags.owndata)  # False: NumPy shares the decoded binary buffer
```

Zero-copy here means **from the decoded buffer to NumPy**. HTTP reception and
MessagePack decoding still allocate memory. The view is read-only when backed
by immutable bytes; use `embedding.copy()` when you need to modify it.

For JSON responses, omit the `Accept` header and decode the base64 buffer first:

```python
import base64

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={"input": ["A mountain lake."], "encoding_format": "tensor"},
)
response.raise_for_status()
tensor = response.json()["data"][0]["embedding"]
buffer = base64.b64decode(tensor["data"])
embedding = np.frombuffer(buffer, dtype="<f4").reshape(tensor["shape"])
```

Base64 decoding allocates a buffer; the resulting NumPy view shares that buffer.

## Token embeddings

Use `/embed_all` with `{"inputs": ["hello"], "encoding_format": "tensor"}`
for supported token-embedding models. The response is a top-level list of tensors,
each shaped `[tokens, dimensions]`. Decode each input independently using the
same `np.frombuffer(...).reshape(...)` pattern. Token counts can differ, so no
batch padding is added. Stacking separate buffers into one array requires a copy.

<details>
  <summary>Tensor wire format</summary>

Each tensor has `shape`, `dtype: "float32"`, `data_encoding` and `data`.
Buffers are contiguous, row-major, little-endian IEEE 754 float32, with exactly
`4 * product(shape)` decoded bytes and no header or padding. IEEE 754 bits,
including signed zero, are preserved. The public dtype is always `float32`,
independent of the backend's weight or compute precision. FP16 and INT8 output
are not currently supported. This is a portable output contract, not a dump of
backend tensor storage: no pointers, device identifiers, strides or uninitialized
allocation padding are transmitted.

JSON uses `data_encoding: "base64"` and a string; MessagePack uses
`data_encoding: "binary"` and bin data. Both `application/msgpack` and
`application/x-msgpack` are accepted.

Pooled vectors have shape `[dimensions]`; token matrices have shape
`[tokens, dimensions]`, preserving returned token order. Empty matrices use `[0, 0]`.
OpenAI-style endpoints place tensors in `data[i].embedding`; `/embed` and
`/embed_all` return a list of tensors. This tensor option is v3-specific.

</details>

## OpenAPI and decoder validation

The live contract is available at `/api-doc/openapi.json` and `/docs`.
OpenAPI uses `Tensor` for JSON (base64 string) and `TensorMessagePack` for
MessagePack (binary data). Both carry the same logical values, shape and dtype;
byte order and row-major layout are fixed by this contract, not negotiated or
inherited from the server CPU. The generated dtype enum lists only implemented
output types. In MessagePack, `format: binary` denotes a **bin** value, not a
string containing base64 or a list of integers.

Requests continue through the existing JSON parser and request validation;
selecting tensor output does not enable uploading or deserializing arbitrary
tensors. Responses use the existing admitted result allocations and bounded
serialization writer. Binary tensors are data, not executable objects; no pickle
or object-construction hooks are involved. JSON and binary decoders still need
resource limits and validation, so neither encoding alone guarantees security.

When consuming a response, validate the supported dtype, rank and nonnegative
integer dimensions, then check `len(decoded_data) == product(shape) * item_size`
with checked arithmetic before creating an array view or allocating output.
Reject unknown dtypes or encodings and malformed base64; bound response sizes
and any decompression before decoding. Do not guess dtype from byte length or
assume buffers have the alignment required by a typed array. In Node.js,
`DataView` reads can explicitly select little-endian without an alignment
assumption.

MessagePack tensor output avoids base64 expansion and per-element numeric
serialization; JSON tensor output remains available for clients preferring
base64. Tensor packing writes little-endian float32 bytes directly into one
admitted contiguous buffer. The byte-order guarantee needs no extra swapping
pass and the response does not contain a second float32 representation. HTTP reception and serialization still copy
bytes; this is not end-to-end zero-copy and no universal latency claim is made.

## Future output precision

Output precision and `encoding_format` are separate choices. `tensor` describes
the container, not a request to expose the model's internal dtype. Future
precision options must be explicitly requested and must preserve today's
float32 default and existing tensor fields. They are not supported request
parameters today.

- `float16` would mean two bytes per element, little-endian IEEE 754 binary16.
  Loss of precision and overflow on conversion must be documented.
- `int8` would mean one signed two's-complement byte per element. A quantized
  embedding also needs a specified quantization scheme, scale granularity and
  zero point (or an explicitly defined symmetric scheme). Integer storage alone
  does not explain how to recover approximate float values or compare vectors.

These are design constraints, not currently advertised dtype enum values.
Each new output type needs matching OpenAPI, Python/Node decoding examples and
wire fixtures before it is enabled. Clients should reject unsupported types,
rather than silently treating every tensor as float32.
