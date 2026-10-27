# Packed embedding tensors

Opt in with `"encoding_format": "tensor"` on `/v1/embeddings`, `/embeddings`,
`/embed` or `/embed_all`. Add `Accept: application/msgpack` for binary data
without base64 expansion. Requests stay JSON; the default `float` output and
existing pooling, normalization and truncation behavior stay unchanged.
Include `model` when serving multiple models. This option is Infinity v3 only.

```bash
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' -H 'Accept: application/msgpack' \
  -d '{"input":["hello"],"encoding_format":"tensor"}' --output embeddings.msgpack
```

## Wire contract

Each tensor contains `shape`, `dtype: "float32"`, `data_encoding` and `data`:

```json
{"shape":[2],"dtype":"float32","data_encoding":"base64","data":"AACAPwAAAMA="}
```

This example represents `[1.0, -2.0]`. Decoded data is contiguous, row-major,
little-endian IEEE 754 float32: exactly `4 * product(shape)` bytes, without
padding or a tensor header. Output dtype is independent of model compute dtype.
JSON uses `data_encoding: "base64"` with a string; MessagePack uses `"binary"`
with a standard bin value. `application/x-msgpack` is also accepted.
`encoding_format` and `Accept` are independent choices.

Pooled vectors have shape `[dimensions]`; token matrices have shape
`[tokens, dimensions]`, preserving token order. `/embed_all` returns one tensor
per input, without batch padding; empty matrices use `[0, 0]`. OpenAI-style
endpoints put tensors in `data[i].embedding`; `/embed` and `/embed_all` return
lists of tensors. The legacy `base64` format keeps native-endian float32;
the portable byte-order guarantee applies to `tensor`.

## Python

Install `requests`, `msgpack` and `numpy`:

```python
import msgpack
import numpy as np
import requests

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers={"Accept": "application/msgpack"},
    json={"input": ["hello"], "encoding_format": "tensor"},
)
response.raise_for_status()
tensor = msgpack.unpackb(response.content, raw=False)["data"][0]["embedding"]
embedding = np.frombuffer(tensor["data"], dtype="<f4").reshape(tensor["shape"])
```

For JSON, omit `Accept`, use `response.json()`, and decode `data` with
`base64.b64decode(tensor["data"], validate=True)` before `np.frombuffer`.
NumPy shares the decoded buffer; HTTP reception, MessagePack unpacking and
base64 decoding still allocate. Use `.copy()` for a writable array.

## Node.js

Install [`@msgpack/msgpack`](https://github.com/msgpack/msgpack-javascript).
Bin values decode to `Uint8Array`; `DataView` handles unaligned buffers and
explicit little-endian reads without copying them.

```js
import { decode } from "@msgpack/msgpack";

const response = await fetch("http://localhost:8000/v1/embeddings", {
  method: "POST",
  headers: { "Content-Type": "application/json", Accept: "application/msgpack" },
  body: JSON.stringify({ input: ["A mountain lake."], encoding_format: "tensor" }),
});
if (!response.ok) throw new Error(`Infinity returned HTTP ${response.status}`);
const result = decode(new Uint8Array(await response.arrayBuffer()));
const tensor = result.data[0].embedding;
if (tensor.dtype !== "float32" || tensor.data_encoding !== "binary" ||
    !(tensor.data instanceof Uint8Array)) {
  throw new Error("Unsupported tensor representation");
}
if (!Array.isArray(tensor.shape) || tensor.shape.length < 1 ||
    tensor.shape.length > 2 ||
    !tensor.shape.every(n => Number.isSafeInteger(n) && n >= 0)) {
  throw new Error("Invalid tensor shape");
}
const byteLength = tensor.shape.reduce((size, n) => size * n, 4);
if (!Number.isSafeInteger(byteLength) || byteLength !== tensor.data.byteLength) {
  throw new Error("Tensor shape and buffer size disagree");
}
const values = new DataView(
  tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength,
);
// Explicit little-endian reads; works even if the bin buffer is unaligned.
console.log(tensor.shape);
if (values.byteLength > 0) console.log(values.getFloat32(0, true));
```

For JSON, use `response.json()` and `Buffer.from(tensor.data, "base64")`.
A direct `Float32Array` view requires alignment and a little-endian host.

## Validation

The live OpenAPI is at `/api-doc/openapi.json` and `/docs`. `Tensor` describes JSON
base64 bytes; `TensorMessagePack` describes MessagePack bin bytes. Each schema
fixes `data_encoding` to the corresponding value.
Validate dtype, rank, nonnegative integer dimensions and decoded byte length
with checked arithmetic before constructing an array. Bound response sizes and
any decompression before decoding untrusted responses; the examples buffer one
response. Requests use the existing JSON parser and memory budgets; selecting
tensor output does not enable arbitrary tensor uploads or executable objects.

MessagePack avoids base64 and per-element numeric encoding. Tensor packing adds
no extra byte-swapping pass or second float32 representation; it is not
end-to-end zero-copy and no universal latency improvement is claimed.

The [Baseten Performance Client](https://github.com/basetenlabs/truss/tree/main/baseten-performance-client)
0.1.16 supports numeric-array responses, but its typed response does not support
this tensor envelope. Use the examples above for tensor output.
