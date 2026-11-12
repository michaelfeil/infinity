# Benchmark method

The [B200 comparison](benchmarks/b200-20261108/results.md) tests 30 checkpoint
candidates against Infinity, vLLM and SGLang. It covers dense embeddings and
reranking, including Qwen3 4B/8B, E5-Mistral 7B, KaLM 12B, Harrier 27B and two
SFR 7B checkpoints. The [pinned manifest](../scripts/benchmark-common-models.json)
records the exact revisions, parameter counts, tasks and input lengths.

Serving a request and producing comparable numerical output are separate
conditions. The report retains failed configurations and diagnostic cases.
Only cases that pass the recorded workload and numerical checks qualify for
three-runtime comparison. These are synthetic serving measurements, not
retrieval, ranking or classification accuracy evaluations.

## Builds and placement

Infinity is a configured **source build** of `3.0.2-beta.2` from the clean
checkout at commit `53dd6f2cb3083e72b65fbe2406913bc09ab75952`. The build uses
CUDA architecture `100` and pinned FA4 AOT and GDN provider locations supplied
through `FA4_ROOT` and `GDN_ROOT`. The published
[results.json](benchmarks/b200-20261108/results.json) `campaign.build`
metadata preserves the recorded portable build command, provider
recipes/manifests, native-library hashes and a content hash of the selected
Torch/native CUDA bundle. This includes `torch/lib` and packaged SDK-local
NVIDIA dependencies. Build commands and provider recipes are operator-provided
provenance declarations, not a standalone cryptographic source-to-binary
attestation. This identifies a configured source build, rather than a published
container image.

The recorded native build command, run from `SOURCE_CHECKOUT/libs/infinity-v3`,
is:

```bash
LIBTORCH="$LIBTORCH" LIBTORCH_CUDA_ARCHITECTURES=100 \
  INFINITY_FA4_ROOT="$FA4_ROOT" INFINITY_GDN_ROOT="$GDN_ROOT" \
  CMAKE_BUILD_PARALLEL_LEVEL=24 CARGO_BUILD_JOBS=24 \
  cargo build --release --locked --no-default-features --features libtorch,http
```

Here `LIBTORCH` is the Torch SDK/CMake prefix used for compilation, distinct
from the runtime library directory `TORCH_LIB`. `FA4_ROOT` and `GDN_ROOT` are
the pinned operator-provided provider locations.

The comparators are vLLM `0.33.0` and SGLang `0.5.23`. Their package files were
checked against the installed wheel records, including missing files. The runner
rechecks package contents, recorded dependency versions, the selected Torch
libraries and Python identities before launching or resuming a campaign.
It selects each serving runtime's own Python environment for executable lookup.
Checkpoint identities include the contents of weight shards and dependency
metadata. Python environment fingerprints cover the interpreter, standard
library and active import roots, including dependency metadata and native files;
derived Python bytecode is excluded. Helper source hashes are also bound.
Inherited library and Python import overrides are removed. Python safe-path
mode excludes implicit working-directory imports; the explicit stock-vLLM
wrapper override remains part of the captured import roots.
vLLM uses Torch `2.14.0`; Infinity
and SGLang use `2.14.1`. Runtime versions and content hashes are included in
the results.

The measured SM100 FA4 provider and its loaded dependencies match the recorded
native-library hashes; their load paths were verified with a live request.
Current tools require complete `infinityfa4/sm_<arch>/` bundles beside the
loaded `libinfinity_torch.so` for every selected GPU architecture. Preserve
those directories and their runtime dependencies when relocating a build.
Opaque build-time fallback paths are rejected before a new campaign starts.

Each checkpoint runs the three runtimes sequentially on the same B200 and
server CPU allocation. Six GPU lanes run concurrently. Each lane has eight
physical server cores and eight separate client cores. GPUs 0, 1, 3, 4, 5 and 6
are used; GPU 2 is outside the serving allocation. Reference inference
uses GPU 7. Other host activity remains uncontrolled.
The recorded server and client allocations use 96 distinct physical cores.
Current tools also reject logical CPU allocations that share a physical core
when the host supplies that topology.

## Workloads and units

All runtimes use BF16, one input or document per request, and JSON float output.
Text cases contain exactly 128 or 512 locally tokenized tokens, including
special tokens. Larger checkpoints also have 2,048-token cases. Reranker lengths
count the query/document pair together. Inputs are generated before timing from
a deterministic pool of 256 samples; no automatic task instruction is added.

Short cases use concurrency 32; 2,048-token cases use concurrency 8. Each case
has three ten-second windows after three seconds of warmup. The batch-token
budget is 16,384. The sequence limit follows the checkpoint, capped at 2,048;
vLLM also honors saved sentence-transformer limits. SGLang's admission limit
reserves one extra position without increasing the actual input length.

Prefix caching and Radix caching are disabled, as is Infinity's Radix MLP.
Infinity CUDA graphs are explicitly enabled; other CUDA graph settings follow
runtime defaults. Dense outputs are L2-normalized. Rerankers return raw logits:
Infinity requests `raw_scores:true`, vLLM requests `use_activation:false`, and
SGLang uses its native raw scores. The client applies no score transformation
to obtain agreement.

Requests/s is successful responses divided by total elapsed window time,
including dispatch, transport, validation and draining in-flight requests.
Server input tokens/s requires complete server usage counts; locally counted
tokens/s is labeled separately. These rates combine the three windows by
summing counts and elapsed time. p50/p95 are medians of the per-window
percentiles, rather than percentiles of pooled latency samples.

The report marks cases whose three window request rates differ by more than
twofold as `timing=variable`. This is a descriptive check of observed variation,
separate from workload validity and numerical eligibility. Flagged
aggregates are retained, with individual
window rates shown; they do not support controlled speedup claims. The cause of
the variation is not established.

The minimum serial latency comes from 20 requests with an 8-token input at
concurrency 1. Startup is one process-start-to-health sample with local weights,
excluding downloads. Storage and kernel cache state is uncontrolled. Isolated,
longer runs are needed for maximum-capacity estimates or small differences.

## Numerical checks and evidence

Every accepted response is checked for valid structure and finite values. Saved
probes must have matching checkpoint revisions, semantic input hashes and output
shapes. Dense probes require cosine similarity at least 0.999 and close norms
(`rtol=0.02`, `atol=0.01`); raw reranker scores use the same value tolerances.
Output order is preserved.

The independent reference uses Hugging Face models in FP32, eager attention,
evaluation mode and inference mode, with TF32 disabled. It uses the same exact
input and the checkpoint's saved CLS, mean or last-token pooling, followed by L2
normalization. Reranker references retain raw logits. Unsupported pipelines or
missing references are reported explicitly; they supply no numerical pass.
These probes are limited numerical checks, not evidence about task quality.
The FP32 run records fresh weight/dependency and Python environment identities
before inference. Accepted reference probes must have the same weight-content
digest as the serving checkpoint. Reference, checkpoint and integrity helper
sources, the client and manifest are identified by content hashes.

The result directory contains a readable report, compact JSON, a CSV and
`evidence.jsonl.gz`. Compressed evidence includes complete saved runtime and
FP32 probe vectors, semantic inputs, per-window counters and latency summaries.
It is a portable subset of the experiment, excluding individual timing samples
and full server logs. Original raw-record hashes are provenance references;
they do not imply that omitted original files can be reconstructed.
`artifacts.json` records the published files' sizes and SHA256 hashes. JSON
`raw_runs` retains complete portable fingerprint inputs and weight-file hashes,
so recorded run fingerprints can be independently recomputed.

## Reproduction

Use separate environments for vLLM, SGLang and the client/reference dependencies.
Native preflight requires `readelf` from binutils and `nvidia-smi` on the host.
Install the client/reference packages from
[`benchmark-requirements.txt`](../scripts/benchmark-requirements.txt) and the
appropriate CPU or CUDA wheel of Torch 2.14.1. Install the serving runtime
versions recorded in the dataset in their own environments. Build Infinity from the
recorded source commit in a separate checkout, following the
[build instructions](../CONTRIBUTING.md#working-in-the-infinity-monorepo) and
the [LibTorch/FA4 backend notes](libtorch-backend.md#infinityfa4), and retain
its provider manifests. The
[FA4 provider build-dependency instructions](../backends/libtorch/cpp/vendor/infinity_fa4/README.md)
document the pinned `build_all.py` recipe; the provider source recipe is in
[`fa4_source.py`](../backends/libtorch/cpp/vendor/infinity_fa4/fa4_source.py).
The benchmark scripts are supplied with this contribution; they are not present
in the earlier benchmarked source commit.
The dataset identifies its measured tool revision in
`campaign.harness_source_commit` and records each helper's content hash.
Use those exact tool bytes to reproduce the published fingerprints. Newer tool
versions can produce new fingerprints even when the serving configuration is
unchanged.

Set `BENCH_OUTPUT` to an absolute output directory, `SOURCE_CHECKOUT` to the
build's repository root, `CLIENT_PYTHON` to the Python interpreter from the
pinned client/reference environment, and `BUILD_METADATA` to the absolute path
of a portable JSON file declaring the recorded build command, provider
recipes/manifests and library hashes consumed by `--extra-metadata`. Set `INFINITY_BIN`,
`TORCH_LIB`, `VLLM_PYTHON` and `SGLANG_PYTHON` to their installed locations.
`TORCH_LIB` is the runtime library directory passed to the server; `LIBTORCH`
is the Torch SDK/CMake prefix used to build it. CPU lanes must fit the machine's
physical topology.

```bash
"$CLIENT_PYTHON" libs/infinity-v3/scripts/benchmark-common.py prepare \
  --output "$BENCH_OUTPUT"

"$CLIENT_PYTHON" libs/infinity-v3/scripts/benchmark-campaign.py \
  --output "$BENCH_OUTPUT" --source-dir "$SOURCE_CHECKOUT" \
  --infinity-bin "$INFINITY_BIN" --torch-lib "$TORCH_LIB" \
  --vllm-python "$VLLM_PYTHON" --sglang-python "$SGLANG_PYTHON" \
  --client-python "$CLIENT_PYTHON" --timezone America/Los_Angeles \
  --extra-metadata "@$BUILD_METADATA" \
  --gpus 0,1,3,4,5,6 \
  --server-cpus 0-7,8-15,24-31,64-71,72-79,80-87 \
  --client-cpus 32-39,40-47,56-63,88-95,96-103,104-111

"$CLIENT_PYTHON" libs/infinity-v3/scripts/benchmark-common.py run \
  --output "$BENCH_OUTPUT" --infinity-bin "$INFINITY_BIN" --torch-lib "$TORCH_LIB" \
  --vllm-python "$VLLM_PYTHON" --sglang-python "$SGLANG_PYTHON" \
  --client-python "$CLIENT_PYTHON" --gpus 0,1,3,4,5,6 \
  --server-cpus 0-7,8-15,24-31,64-71,72-79,80-87 \
  --client-cpus 32-39,40-47,56-63,88-95,96-103,104-111

env -u LD_LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT -u PYTHONPATH -u PYTHONHOME \
  PYTHONSAFEPATH=1 CUDA_VISIBLE_DEVICES=7 taskset -c 112-119 "$CLIENT_PYTHON" \
  libs/infinity-v3/scripts/benchmark-fp32-reference.py \
  --input "$BENCH_OUTPUT" --output "$BENCH_OUTPUT/fp32-reference.json" \
  --device cuda:0

"$CLIENT_PYTHON" libs/infinity-v3/scripts/report-embedding-benchmarks.py \
  --input "$BENCH_OUTPUT" --reference "$BENCH_OUTPUT/fp32-reference.json" \
  --output "$BENCH_OUTPUT/report"
```

The recorded reference used GPU 7 and CPU affinity `112-119`. A CPU-only
reference remains a valid numerical check but is a different recorded
environment and should not be described as an exact reproduction. The exporter
requires all manifest/runtime outcomes by default. `--allow-incomplete` creates
a report labeled provisional. A different source build, checkpoint revision,
hardware allocation or method constitutes a new experiment and gets new
fingerprints.
