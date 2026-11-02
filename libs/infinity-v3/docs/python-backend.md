# Python backend

The `python-backend` backend lets an external Python package supply model execution
while Infinity continues to own the HTTP API, tokenization, batching, queues,
metrics, and result encoding. The backend is selected at runtime; no Rust code
changes are needed to add a new Python plugin.

## Build and selection

Build the router with the `python-backend` feature:

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
  Admission uses the full validated value; the deferred completion queue is
  capped separately.

All other keys are passed to the plugin as opaque configuration. `device` is a
common convention but is not interpreted by the frontend.

`INFINITY_PYTHON_CONFIG` is the environment equivalent of `--python-config`.
`INFINITY_PYTHONPATH` can add import paths for source checkouts.

The frontend pads each batch to its longest sequence. `input_ids`,
`attention_mask`, `token_type_ids`, and `position_ids` are flattened row-major
arrays that reshape to `[batch_size, seq_len]`. The mask is 1 for real tokens
and 0 for padding; `lengths` contains the unpadded length of each sequence.
Input arrays are numpy views over Rust-owned buffers; the payload keeps those
buffers alive while the request is queued. Treat the arrays as read-only.
The plugin owns model loading, device placement, and model-specific preprocessing.
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
`float32`, C-contiguous, and shaped `[rows, width]` with `width > 0`:

- `embed`: one row per pooled index, followed by the real token rows for each
  raw index in request order.
- `predict`: one row per sequence.

`set_result` snapshots the values into Rust-owned memory before returning, so
plugins can then reuse their output buffers. The snapshot is admitted against
the host result memory budget before allocation. Invalid or over-budget outputs
raise an exception without completing the response; the executor must deliver
that error with `set_exception`.

`response.set_exception(error)` completes the request with an error. Both
completion methods release the in-flight permit even if the plugin retains the
response object. The executor must not block the calling Rust worker; a common implementation uses one
dedicated Python thread and a bounded queue. `close` must join that thread and
release model resources.

## Lifecycle and threading

- One executor thread per process is recommended. The Rust backend already
  serializes a single model worker and completion worker.
- The Rust backend bounds outstanding forwards with `queue_depth`. The deferred
  completion queue is bounded by `min(queue_depth, 256)`, so the model worker
  can keep accepting coalesced batches without preallocating an oversized
  channel. Other backend kinds retain their completion depth of 2.
- `healthy` is called by the shared health endpoint.
- The Python interpreter stays initialized for the process lifetime. The plugin
  must be importable before the first model is created.
- Do not fork after initializing device runtimes. Use a fresh process per
  device.

## Buffer ownership

- **Rust to Python:** input arrays are borrowed numpy views over the Rust
  `PaddedInputs` buffers. A Python owner object keeps the `Arc` alive while the
  executor queue holds the payload.
- **Python to Rust:** `set_result` validates the array and copies its values into
  one Rust-owned snapshot. Making a numpy view read-only cannot prevent writes
  through its base array, so Python-owned output memory is never shared with
  Rust readers after completion.
- Embedding rows share the immutable Rust snapshot through `SharedF32` and
  `bytes::Bytes`; packing and response encoding do not copy individual rows.
- Classifier predictions own `Vec<f32>` values. Their row copies reserve space
  before allocation, including the overlap with the snapshot.
- Post-processing that mutates shared embeddings, such as normalization,
  reserves and retains the additional memory before copying. Dimension
  truncation only narrows the view.
- Zero-copy inputs apply only in-process. A separate Python worker process needs
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
