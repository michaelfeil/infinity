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
  --python-config '{"backend":"my-backend","device":"auto"}'
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

All other keys are passed to the plugin as opaque configuration. `device` is a
common convention but is not interpreted by the frontend.

`INFINITY_PYTHON_CONFIG` is the environment equivalent of `--python-config`.
`INFINITY_PYTHONPATH` can add import paths for source checkouts.

The frontend pads each batch to its longest sequence. `input_ids`,
`attention_mask`, `token_type_ids`, and `position_ids` are flattened row-major
arrays that reshape to `[batch_size, seq_len]`. The mask is 1 for real tokens
and 0 for padding; `lengths` contains the unpadded length of each sequence.
Input arrays are read-only numpy views over Rust-owned buffers; the payload
keeps those buffers alive while the request is queued. Writing to a view or
re-enabling its write flag is rejected. Plugins that need in-place preprocessing
must first copy the affected arrays into their own storage.
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
- `serving_limits`: resolved `max_input_length` (effective sequence length),
  `max_batch_tokens`, and `max_batch_requests` (`None` when uncapped)

`serving_limits` is always supplied as a mapping before warmup. Factories must accept
that keyword or `**config`. The same shared limits are available to native
backends; they describe existing frontend bounds.

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
response object. Core wraps the plugin in `ThreadedExecutor`, so `submit`
runs on its persistent Python pool thread; plugins may execute synchronously
there or retain the envelope for deferred completion. Plugins that manage their
own asynchronous work must drain it in `close` and release model resources.

### Optional replica capability discovery

The selected plugin module may export a `capabilities(config_json)` hook returning
exactly `visible_device_count` and `supports_replication`:

```python
def capabilities(config_json):
    return {"visible_device_count": 4, "supports_replication": True}
```

The hook receives the same `config_json` as the factory and runs synchronously
before replica placement and model startup. It may run more than once, so keep
it cheap and side-effect free. The count must be a positive integer and the
flag must be a boolean; invalid values fail startup clearly.

Protocol-1 plugins without the hook keep the single-device defaults
(`1`, `False`). Existing `--device-id` and `--backend-device-ids` selection
then applies per replica: `auto` expands only when replication is supported,
explicit ordinals must be visible, and each replica receives its own
`device_id`. `BackendPool` schedules batches across replicas.

## Lifecycle and threading

- Capability discovery is synchronous. Each factory runs on its replica's
  persistent Python executor thread, also used for inference, health, and cleanup.
  Rust waits on a startup envelope with the GIL released; startup errors surface
  directly. Replicas can initialize concurrently, and the server waits for all
  factories and warmups before serving. Factory arguments, including
  `serving_limits`, are unchanged. Factories remain synchronous Python functions;
  pure Python initialization still shares the GIL.
- `infinity_v3_python.threaded.ThreadedExecutor` wraps the plugin with one
  persistent `ThreadPoolExecutor(max_workers=1)` per backend replica. It owns
  the plugin's single execution thread.
- `submit` schedules plugin work on that pool and returns immediately, so the
  Rust backend worker does not block on a forward pass. Plugin exceptions are
  caught on the pool thread and complete the Rust response envelope through
  `set_exception`. Legacy plugins that internally defer completion remain valid.
- `ThreadedExecutor` uses one state lock and a busy flag to enforce single-flight:
  a second submission raises a clear busy/overload error instead of waiting.
  The busy flag spans the whole response envelope, including legacy plugins that
  complete later, and is released once after successful completion, exception
  delivery, or scheduling failure. The Rust bridge also admits one in-flight
  envelope per replica, while deferred responses retain the replica lease until
  completion. This is invariant defense, not a second pending queue.
  `healthy` is serialized through the same pool. `close` is the final pool task,
  after accepted work, so cleanup uses the execution thread. Cleanup failures
  are reported; repeated calls to `close` are safe.
- One shared frontend queue and scheduler per model hands batches to exclusive
  `BackendPool` replica leases; there is no separate frontend per device.
  Distinct replicas have distinct Python pool threads. Python bytecode still
  serializes under one interpreter GIL, while waits and native operations may
  release it.

## Buffer ownership

- **Rust to Python:** input arrays are borrowed, read-only numpy views over the
  Rust `PaddedInputs` buffers. A Python owner object keeps the `Arc` alive while
  the executor queue holds the payload. Derived views inherit read-only access;
  plugins must copy arrays before mutating them.
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
