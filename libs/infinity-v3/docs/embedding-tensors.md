# Packed embedding tensors

Set `"encoding_format": "tensor"` on `/embeddings`, `/v1/embeddings`, `/embed`,
or `/embed_all`. Requests remain JSON; existing float arrays and base64 outputs
are unchanged. This option applies to V3, not the Python server.

Each tensor contains `shape`, `dtype: "float32"`, `data_encoding`, and `data`.
Data is contiguous, row-major, little-endian IEEE 754 float32: exactly
`4 * product(shape)` bytes, with no padding or header. Int8 is deferred.

- JSON responses use `data_encoding: "base64"` and a base64 string.
- `Accept: application/msgpack` (or `application/x-msgpack`) uses
  `data_encoding: "binary"` and MessagePack **bin** data.
- Pooled vectors have shape `[dimensions]`. `/embed_all` matrices have shape
  `[tokens, dimensions]`, with one independent buffer per input and no batch
  padding. Returned token order is preserved; empty matrices have shape `[0, 0]`.

OpenAI-style responses place tensors in `data[i].embedding`; `/embed` and
`/embed_all` return a list of tensors. Normalization, truncation, and pooling
retain their existing behavior.

```python
import base64, msgpack, numpy as np, requests

r = requests.post("http://localhost:7997/v1/embeddings",
    headers={"Accept": "application/msgpack"},
    json={"input": ["hello"], "encoding_format": "tensor"})
r.raise_for_status()
t = msgpack.unpackb(r.content, raw=False)["data"][0]["embedding"]
buffer = base64.b64decode(t["data"]) if t["data_encoding"] == "base64" else t["data"]
embedding = np.frombuffer(buffer, dtype="<f4").reshape(t["shape"])
```

For JSON, use `r.json()` instead of `msgpack.unpackb`. For `/embed_all`, send
`{"inputs": ["hello"], "encoding_format": "tensor"}` and decode each top-level
tensor. Include the served `model` name when multiple models are deployed.
