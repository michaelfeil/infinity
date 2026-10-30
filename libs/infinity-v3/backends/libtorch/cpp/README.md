# Native C++ backend layout

- `models/`: model interface, model families, embedding and decision heads.
- `helpers/`: configuration and weights, shared attention composition, FP8 linear wrapper, image-feature cache.
- `batching/`: packed input and media descriptors, radix folding, CUDA graph lifecycle, pooling interface.
- `kernels/`: CUDA operators, device arithmetic and their native entry points.
- `tests/`: existing native regression fixtures and reference data.
- `vendor/`: pinned third-party sources and their licenses.
- `bridge.cpp` and `bridge.h`: C ABI, model construction and request dispatch.

Includes use paths relative to this directory, so dependencies between categories
are explicit. `models/model.h` defines the model contract; packed request data
belongs to `batching/packed_input.h`, options and checkpoint weights to
`helpers/options.h`, and shared attention composition to `helpers/attention.h`.

The CMake target and test names remain stable. CUDA sources keep their original
compile flags and vendored kernel includes. BERT's existing implementation is
in `models/bert_model.h`; its public configuration remains `TeiConfig` from the
C ABI.

## Where to start

- **Adding a model:** read [Model](models/model.h), then the family selection in
  `Engine::ready` in [bridge.cpp](bridge.cpp). Configuration and checkpoint loading
  start in [the Rust adapter](../src/lib.rs). The existing
  [encoder](../tests/encoder_parity.rs) and [decoder](tests/decoder_models_test.cpp)
  parity fixtures exercise that loading path as well as inference.
- **Changing batching:** start with [core's queue actor](../../../core/src/queue_actor.rs)
  and its inline tests for client grouping, limits and cancellation. Then follow
  [PackedInput](batching/packed_input.h) into [radix folding](batching/radix.cpp).
  [radix_test.cpp](tests/radix_test.cpp) shows which row mappings are valid.
- **Debugging incorrect outputs:** follow `forward_radix` in [bridge.cpp](bridge.cpp)
  through model execution and pooling, then `pack_output` in
  [the Rust adapter](../src/lib.rs). For premature reuse or cancellation problems,
  read [OutputLease](batching/output.h) and [output_test.cpp](tests/output_test.cpp):
  a model can accept another submission while the previous output is still owned
  by the completion worker.
