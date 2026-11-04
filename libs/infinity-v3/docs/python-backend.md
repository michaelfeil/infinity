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
- `serving_limits`: optional resolved router limits. This keyword is passed
  only when the factory declares an explicit named `serving_limits` parameter
  (a `**kwargs` catch-all does not opt in, so legacy factories keep their old
  arguments). The value is a mapping with `max_input_length`, `max_batch_tokens`
  and `max_batch_requests` (`None` when request count is uncapped). It is
  resolved metadata, not duplicated user configuration.

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
response object. By default Core wraps the plugin in `ThreadedExecutor`, so
`submit` runs on its persistent Python pool thread; plugins may execute
synchronously there or retain the envelope for deferred completion. A plugin
that advertises `synchronous_submit` is not wrapped and must complete the
envelope inside `submit`. Plugins that manage their own asynchronous work must
drain it in `close` and release model resources.

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

## Optional capabilities

An executor may also implement `capabilities()` after initialization. Executors
without the method keep the legacy behavior. The method returns a mapping:

```python
class MyExecutor:
    def capabilities(self):
        return {
            "protocol_version": 1,
            "synchronous_submit": True,
            "packed_limits": {
                "max_total_tokens": 32768,
                "max_sequences": 64,
                "max_seq_len": 32768,
            },
        }
```

`protocol_version` is optional; when present it must be `1`.
`synchronous_submit` is optional and defaults to `false`. When it is `true`,
Core reads it from the raw plugin once at startup and does not wrap the plugin
in `ThreadedExecutor`; `submit` then runs directly on the serialized Rust
backend worker and must complete the response before returning. This opt-in
adds no Python request queue, worker thread, or deferred buffering. The Rust
`BackendThread` still serializes submissions, and `BackendPool` keeps the
queued replica lease busy until completion. Plugins that omit the field, or set
it to `false`, keep the legacy threaded wrapper. The value must be a plain
boolean; any other value, including `null`, is a startup error.
`packed_limits` is optional, but when present all three keys are required and
each value must be a plain positive integer (not a boolean). The values describe
the real capacity of one dispatched call:

- `max_total_tokens`: maximum total tokens across the batch.
- `max_sequences`: maximum sequences per call. Rust uses this through the
  existing `max_batch_size` cap, including when `--max-batch-requests` is unset.
- `max_seq_len`: maximum tokens in any one sequence.

At startup Rust computes the largest batch it can actually form:

```text
effective_requests = min(max_batch_requests or max_sequences, max_sequences)
reachable_total_tokens = min(max_batch_tokens, effective_requests * max_input_length)
```

It rejects the plugin when `reachable_total_tokens > max_total_tokens` or
`max_input_length > max_seq_len`; either is an actionable startup error, never a
silent truncation. The CLI flag for the resolved `max_input_length` is
`--max-sequence-length`; `max_input_length` is the internal field name.
`max_total_tokens` is the logical valid token capacity, not a physical rounded
execution bucket: if the physical bucket is larger than the
valid cap, advertise the logical cap and use a separate field for the physical
size. Advertise the full context the plugin can actually execute. Malformed
capabilities are startup errors. Omitting capabilities leaves batching and
limits unchanged.

## Lifecycle and threading

- Backend startup and capability discovery are synchronous. The plugin factory
  loads the model before the backend returns; startup errors surface directly.
- By default `infinity_v3_python.threaded.ThreadedExecutor` wraps the plugin
  with one persistent `ThreadPoolExecutor(max_workers=1)` per backend replica.
  It owns the plugin's single execution thread.
- Default `submit` schedules plugin work on that pool and returns immediately,
  so the Rust backend worker does not block on a forward pass. Plugin exceptions
  are caught on the pool thread and complete the Rust response envelope through
  `set_exception`. Legacy plugins that internally defer completion remain valid.
- An explicitly synchronous plugin (`capabilities()['synchronous_submit'] is
  True`) is not wrapped. Its `submit` executes inline on the serialized Rust
  backend worker, completes the envelope before returning, and never creates a
  Python request queue or background thread. Missing or false keeps the default
  threaded behavior.
- Queued work is bounded by the existing Rust `queue_depth` admission permits.
  `healthy` is serialized through the same pool. `close` is the final pool task,
  after queued work, so cleanup uses the execution thread. Cleanup failures are
  reported; repeated calls to `close` are safe.
- One shared frontend queue and scheduler per model hands batches to exclusive
  `BackendPool` replica leases; there is no separate frontend per device.
  Distinct replicas have distinct Python pool threads. Python bytecode still
  serializes under one interpreter GIL, while waits and native operations may
  release it.

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
