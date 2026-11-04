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
response object. Core invokes `submit` on a persistent worker, so plugins may
execute synchronously there or retain the envelope for deferred completion.
Plugins that manage their own asynchronous work must drain it in `close` and
release model resources.

### Optional replica capability discovery

A plugin can report replica capabilities before the frontend chooses devices.
Export a module-level `capabilities` hook (or a `capabilities` attribute on the
factory) that accepts the same `config_json` string passed to the factory and
returns `BackendCapabilities`:

```python
from infinity_v3_python import BackendCapabilities

def capabilities(config_json):
    return BackendCapabilities(
        visible_device_count=4,
        supports_replication=True,
    )

def create_executor(**config):
    return MyExecutor(config)
```

`visible_device_count` must be a positive integer and `supports_replication`
must be a boolean. Invalid, malformed, or zero capabilities fail startup with a
clear error. The hook runs before replica placement and before model weights are
loaded, may run more than once, and must be cheap and side-effect free. Core
awaits discovery on a blocking worker so a synchronous hook does not occupy the
async router thread; plugins do not need to implement a Python coroutine.

Protocol-1 plugins that omit the hook keep the single-device defaults
(`visible_device_count=1`, `supports_replication=False`), so existing plugins
remain compatible without changes. With discovery enabled, existing
`--device-id` and `--backend-device-ids` selection applies per replica:
`auto` expands to every visible device only when replication is supported, an
explicit list selects those ordinals, and a single selected device must be
within the reported visible count. Each replica is a separate backend worker
with its own `device_id`, and `BackendPool` schedules batches across them.

## Lifecycle and threading

- Initial startup is synchronous: backend construction waits until the plugin
  factory finishes loading the model or returns a startup error. The factory
  runs on the persistent worker that will execute its requests.
- There is one shared frontend queue and scheduler per model. It hands each
  batch to an exclusive replica lease in `BackendPool`; it does not create a
  separate frontend per device.
- Each backend replica owns one persistent Python worker thread and one Rust
  command queue. Submission validates and enqueues the Rust payload and
  completion slot without taking the GIL, then returns the normal deferred
  receiver immediately, even when the plugin's `submit` is synchronous. The
  worker takes the GIL only around executor construction, health checks,
  `submit`, and `close`.
- The worker serializes plugin submissions for its replica while waiting on the
  Rust queue without the GIL. Completion packing and result delivery stay on
  the separate Rust completion path. Distinct replicas have distinct workers,
  so Python waits or native operations that release the GIL can overlap across
  replicas; Python bytecode itself still serializes under one interpreter GIL.
- The Rust backend bounds pending calls with admission permits equivalent to
  `queue_depth`. The command channel uses lazy allocation, so its request
  backlog cannot exceed those admitted calls and a large configured depth does
  not preallocate a large buffer. Overload rejection releases its admission
  permit.
- `healthy` is routed through the same replica worker, preserving serialized
  executor ownership.
- Shutdown stops admission, drains queued work and settles abandoned callers,
  closes the executor, and joins the worker without holding the GIL. No
  per-request threads are spawned.
- The Python interpreter stays initialized for the process lifetime. The plugin
  must be importable before the first model is created.
- Do not fork after initializing device runtimes. Use a fresh process per
  device unless the plugin explicitly supports sharing one process.

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
