# Python backend

The `python` backend lets an external Python package supply model execution
while Infinity continues to own the HTTP API, tokenization, batching, queues,
metrics, and result encoding. The backend is selected at runtime; no Rust code
changes are needed to add a new Python plugin.

## Build and selection

Build the router with the `python` feature:

```bash
cargo build --release -p infinity_v3-router --no-default-features --features python-backend,http
```

Run with:

```bash
infinity_v3 \
  --model-id /models/all-MiniLM-L6-v2 \
  --backend python-backend \
  --python-config '{"backend":"my-backend","device":"auto","queue_depth":32}'
```

`--python-config` is a JSON object. Repeat it per model when serving multiple
models:

```bash
--python-config '{"device":"cpu"}' \
--python-config '{"device":"gpu","backend":"other-backend"}'
```

Reserved keys:

- `backend`: entry-point name in `infinity_v3.backends` or a
  `module:attribute` factory. If omitted, the sole installed entry point is
  used.
- `queue_depth`: positive integer maximum in-flight forwards. Defaults to 32.

All other keys are passed to the plugin as opaque configuration. `device` is a
common convention but is not interpreted by the frontend.

`INFINITY_PYTHON_CONFIG` is the environment equivalent of `--python-config`.
`INFINITY_PYTHONPATH` can add import paths for source checkouts.

The frontend always sends packed token IDs, token types, positions, sequence
lengths, pooling mode, and requested output indices. Input arrays are numpy
views over Rust-owned buffers; the payload keeps those buffers alive while the
request is queued. Treat the input arrays as read-only. The plugin is responsible
for padding, model loading, device placement, and model-specific preprocessing.
The response is one contiguous `float32` numpy array.

## Plugin contract

Install the `infinity-v3-python` package. It provides the protocol constants and
the discovery helpers:

```bash
pip install libs/infinity-v3/python
```

For source checkouts, `INFINITY_PYTHONPATH=libs/infinity-v3/python` is enough
while developing the plugin.

An entry point points at a factory:

```toml
[project.entry-points."infinity_v3.backends"]
my-backend = "my_backend.plugin:create_executor"
```

The module must export `PROTOCOL_VERSION` matching `infinity_v3_python`:

```python
from infinity_v3_python.api import PROTOCOL_VERSION

def create_executor(**config):
    return MyExecutor(config)
```

The factory receives:

- `model_path`: local model directory
- `dtype`: selected dtype string
- `model_type`: `embedding` or `classifier`
- `pooling`: `cls`, `mean`, `last_token`, or `splade`
- `device_id`: replica/device ordinal
- `config_json`: the `--python-config` JSON string
- `queue_size`: the resolved `queue_depth`

The returned executor must implement:

```python
class MyExecutor:
    def submit(self, route, response, payload): ...
    def healthy(self) -> bool: ...
    def close(self) -> None: ...
```

`route` is `embed` or `predict`. `payload` is a dictionary of numpy arrays and
integers:

```text
batch_size, seq_len, input_ids, attention_mask,
token_type_ids, position_ids, lengths,
pooled_indices, raw_indices
```

`response.set_result(array)` completes the Rust future. The array must be
`float32`, C-contiguous, and shaped `[rows, width]`:

- `embed`: one row per pooled index, followed by the real token rows for each
  raw index in request order.
- `predict`: one row per sequence.

`response.set_exception(error)` completes the request with an error. The
executor must not block the calling Rust worker; a common implementation uses one
dedicated Python thread and a bounded queue. `close` must join that thread and
release model resources.

## Lifecycle and threading

- One executor thread per process is recommended. The Rust backend already
  serializes a single model worker and completion worker.
- The Rust backend bounds outstanding forwards with `queue_depth`.
- `healthy` is called by the shared health endpoint.
- The Python interpreter stays initialized for the process lifetime. The plugin
  must be importable before the first model is created.
- Do not fork after initializing device runtimes. Use a fresh process per
  device.

## Zero-copy transfer

- **Rust to Python:** input arrays are borrowed numpy views over the Rust
  `PaddedInputs` buffers. A Python owner object keeps the `Arc` alive while the
  executor queue holds the payload.
- **Python to Rust:** the envelope carries `Py<PyArrayDyn<f32>>` and validates
  C-contiguity before sending it. Embedding rows become `SharedF32` values
  backed by a `bytes::Bytes` owner that keeps the numpy allocation alive; no
  row copy occurs through the response path.
- Classifier predictions still own `Vec<f32>` values.
- Post-processing that mutates embeddings, such as normalization, copies the
  shared storage once on demand. Dimension truncation only narrows the view.
- Zero-copy applies only in-process. A separate Python worker process needs
  shared memory or another IPC mechanism.

## Testing

The OSS package includes a deterministic CPU executor:

```bash
--backend python-backend \
--python-config '{"backend":"infinity_v3_python.testing:create_executor"}'
```

Use it for HTTP, batching, envelope, and lifecycle tests. Set
`INFINITY_PYTHONPATH` when running from a source checkout; installed packages
need no path override.
