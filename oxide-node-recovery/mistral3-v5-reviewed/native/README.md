# Native b66 Mistral3 packet (source-only handoff)

This directory contains the CPU-only metadata control and the Root packet for
`nvidia/Nemotron-3.5-Rerank-8B-BF16`, a `Mistral3ForSequenceClassification`
checkpoint with a Pixtral vision tower and a Ministral3 text encoder.

Nothing here was compiled, linked, loaded, or run on a GPU. The CPU control was
run exactly once with `CUDA_VISIBLE_DEVICES=''` and read only `config.json`, the
safetensors index, and safetensors headers. It did not import Torch, load
`libtei_torch.so`, create a CUDA context, query a GPU, use the network, or read a
tensor payload.

## Files

| Path | Purpose |
| --- | --- |
| `native_b66_control.py` | Pure-stdlib CPU metadata control and shape validation |
| `root_paired_controller.py` | Root-only paired Native-gold/Oxide-candidate/lifecycle/perf controller |
| `original-config.json` | Byte-identical copy of the checkpoint `config.json` |
| `tensor-inventory.json` | Full 531-tensor inventory with shard/dtype/shape/offsets |
| `CPU-control.json` | Machine-readable control report |
| `native-b66-cpu-control.log` | Log from the required CPU-control run |
| `include/mistral3_b66.h` | Fixed-width C packet ABI, no C++/Torch/CUDA types |
| `src/mistral3_runner.cpp` | Real C++ packet/lifecycle/reference/perf implementation |
| `consumer/Cargo.toml` | Rust consumer manifest; std + libloading only |
| `consumer/src/main.rs` | Typed-error Rust consumer with lifecycle and perf loop |

## CPU control (already run)

```sh
cd /tmp/oxide-all-model-feature-matrix-v1/mistral3-v2/native
CUDA_VISIBLE_DEVICES='' python3 native_b66_control.py \
  > /tmp/oxide-all-model-feature-matrix-v1/mistral3-v2/native/native-b66-cpu-control.log 2>&1
```

Expected summary: 531 tensors across 4 shards; categories vision 218,
projector 4, language 308, score 1, retained 531; all 531 expected-shape checks
pass; SDK SHA-256 begins `b6607c`.

## How the original config and tensors are loaded later

1. `original-config.json` is the checkpoint `config.json` byte-for-byte. The
   runner and Rust consumer parse it, flatten nested keys with dots (for example
   `text_config.hidden_size` and `architectures.0`), and pass each pair through
   the frozen `tei_option` entry point before `tei_ready`.
2. `original-config.json` is never flattened or rewritten: the runner parses the
   nested `rope_parameters` (YaRN) and `llama_4_scaling` objects in memory and
   emits `tei_option` keys only in memory. The on-disk config and checkpoint are
   byte-hash checked before and after the Root run.
3. `model.safetensors.index.json` names all 531 tensors and 4 shards. For every
   shard the runner/consumer reads only the 8-byte little-endian header length
   and the JSON header, then builds ordered `TeiWeight`/`NativeWeight`
   descriptors. Offsets are relative to the header payload start
   (`8 + header_bytes`), matching `data_offsets`.
4. The shard file descriptor is passed to `tei_load_safetensors` with
   `data_start`, `data_length`, all descriptors, worker/buffer settings, and
   `direct_io=0`. The native reader duplicates the descriptor and drains every
   read before returning, so the host may close it after the call.
5. `tei_ready` builds the Mistral3 model. Reference cases then call
   `tei_forward` (mean pool and classification score) and the deferred lifecycle
   calls `tei_forward_submit` -> `tei_output_wait` -> `tei_output_destroy`.
   `tei_destroy` releases the engine.

## Root build commands (later, inside the GPU/Torch image)

The packet header and runner are source-only. Root provides the compiler, Torch
and CUDA include/link paths, and the frozen `libtei_torch.so`. The SDK is:

```text
/tmp/infinity-fast-weight-reader/libs/infinity-v3/target/debug/build/infinity-backend-libtorch-0ee14ee2449a0322/out/build/libtei_torch.so
sha256 b6607c0025b1bc65d7a2e56951ab28d0c25871da8e7620565f54b9da15d92bd4
```

Build the standalone runner (add Root's Torch/CUDA link flags as needed):

```sh
cd /tmp/oxide-all-model-feature-matrix-v1/mistral3-v2/native
g++ -std=c++20 -O2 -fPIC -Iinclude \
  src/mistral3_runner.cpp \
  -L/tmp/infinity-fast-weight-reader/libs/infinity-v3/target/debug/build/infinity-backend-libtorch-0ee14ee2449a0322/out/build \
  -ltei_torch \
  -Wl,-rpath,/tmp/infinity-fast-weight-reader/libs/infinity-v3/target/debug/build/infinity-backend-libtorch-0ee14ee2449a0322/out/build \
  -o mistral3_b66_runner
```

Build the same file as a shared packet library instead:

```sh
g++ -std=c++20 -O2 -fPIC -shared -DMISTRAL3_B66_NO_MAIN -Iinclude \
  src/mistral3_runner.cpp \
  -L/tmp/infinity-fast-weight-reader/libs/infinity-v3/target/debug/build/infinity-backend-libtorch-0ee14ee2449a0322/out/build \
  -ltei_torch -o libmistral3_b66.so
```

Build the Rust consumer from the offline cache:

```sh
cd /tmp/oxide-all-model-feature-matrix-v1/mistral3-v2/native/consumer
CARGO_NET_OFFLINE=true cargo build --release --offline
```

## Paired gold / candidate / lifecycle / perf controller

`root_paired_controller.py` runs the Native b66 SDK runner (gold) and the fresh
optimized Oxide `mistral3_vectors` candidate in separate processes, pairs the
deterministic text/image/score cases, checks repeatability, then records
lifecycle and profile-window timings. It refuses to run unless
`CUDA_VISIBLE_DEVICES` is set and `INFINITY_OXIDE_ROOT_GPU0_LOCKED=1`, and it
never rewrites the checkpoint. Offline preflight is safe:

```sh
CUDA_VISIBLE_DEVICES='' python3 root_paired_controller.py --preflight-only \
  --checkpoint /workspace/model-performance/michaelfeil/benchmark-overview/models/nvidia--Nemotron-3.5-Rerank-8B-BF16
```

Root GPU0:

```sh
CUDA_VISIBLE_DEVICES=0 INFINITY_OXIDE_ROOT_GPU0_LOCKED=1 \
  python3 root_paired_controller.py \
  --checkpoint /workspace/.../nvidia--Nemotron-3.5-Rerank-8B-BF16 \
  --native-runner /path/mistral3_b66_runner \
  --candidate-bin /path/mistral3_vectors \
  --fa4-root /path/fa4 --out /tmp/mistral3-paired-fresh \
  --iterations 20 --warmup 3
```

`--emit-vectors DIR` on the Native runner writes raw little-endian f32 case
vectors. The Oxide candidate writes `*.pooled.f32`, `*.raw.f32` and
`*.logits.f32` in `INFINITY_MISTRAL3_OUT_DIR`.

## Root run commands (later, on a real GPU)

The runner and consumer refuse to run while `CUDA_VISIBLE_DEVICES` is empty or
unset. Root must override the CPU-only guard explicitly, for example:

```sh
export CUDA_VISIBLE_DEVICES=0
cd /tmp/oxide-all-model-feature-matrix-v1/mistral3-v2/native
./mistral3_b66_runner \
  --checkpoint /workspace/model-performance/michaelfeil/benchmark-overview/models/nvidia--Nemotron-3.5-Rerank-8B-BF16 \
  --device cuda:0 --iterations 20 --warmup 3
```

```sh
export CUDA_VISIBLE_DEVICES=0
cd /tmp/oxide-all-model-feature-matrix-v1/mistral3-v2/native/consumer
export LD_LIBRARY_PATH=/tmp/infinity-fast-weight-reader/libs/infinity-v3/target/debug/build/infinity-backend-libtorch-0ee14ee2449a0322/out/build:${LD_LIBRARY_PATH}
./target/release/mistral3-b66-consumer \
  --library /tmp/infinity-fast-weight-reader/libs/infinity-v3/target/debug/build/infinity-backend-libtorch-0ee14ee2449a0322/out/build/libtei_torch.so \
  --checkpoint /workspace/model-performance/michaelfeil/benchmark-overview/models/nvidia--Nemotron-3.5-Rerank-8B-BF16 \
  --device cuda:0 --iterations 20 --warmup 3
```

Reference cases use deterministic synthetic token IDs and synthetic float pixel
rows; they exercise the image+text packing, `image_token_index=10`, merge size 2,
patch size 14, mean pooling, classification score, and the deferred lease
lifecycle. They are not trained-quality fixtures. The paired controller compares
the Native and Oxide outputs and records the actual max-abs/cosine deltas; it
does not assume parity.

## Pending

- Root compile and real-GPU execution of the C++ runner and Rust consumer.
- Numerical parity against public Transformers 5.12.1 fixtures or trained
  reference outputs (the current CPU control is metadata-only).
- Real tokenizer / Pixtral image preprocessing parity; the packet cases use
  synthetic IDs and pixels.
- Any quality, retrieval, ranking, safety, or performance claim.

## Non-claims

- No Torch import, native library load, CUDA context, device query, network use,
  or tensor payload read occurred in the CPU control.
- No C++ or Rust source in this directory was compiled or executed.
- No numerical parity, throughput, latency, or model-quality claim is made.
- The shape validation checks safetensors metadata only; it does not validate
  tensor values.
