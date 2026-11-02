# Mistral3 / Pixtral / Ministral3 Oxide composer v6 WIP (current Root base)

Private composition for the cached `nvidia/Nemotron-3.5-Rerank-8B-BF16`
`Mistral3ForSequenceClassification` checkpoint. This packet owns the Mistral3 /
Pixtral multimodal wrapper and the Ministral3 Transformers-text recipe only. The
generic Qwen3 dense, Qwen3 reranker and plain-Mistral H4096 shared decoder stay
owned by the `parity_catalog` dense-family worktree; this packet reuses its
frozen plain-Mistral math contract instead of redefining it.

Base: current `/tmp/infinity-plain-cuda` Root working tree at public
`55380b7` plus the 625DenseV5 + Qwen35 e3c38 + MeanV6 decoder-fix six-file
overlay (`cublas.rs`, `decoder.rs`, `decoder_model.rs`, `lib.rs`,
`native_rope_tables.rs`, `decoder_classifier.rs`). This packet is a source-only V6 WIP
composer awaiting the V2 independent review observations; it is not sealed: it adds Mistral3/Minstrall3 on top of that base and does not write or
reset the Root worktree. The upstream V4 patch is not applied; V5 positive Llama
window rejection and V6 BF16 mean <=65535 / F32 fallback >65535 are preserved.
No public push, CUDA context, device compile, Torch, Docker, network, NVIDIA
query, or tensor payload read was performed while authoring this packet.

## Actual checkpoint composition

The original config and 531 safetensors headers are retained byte-for-byte:

| Category | Tensors |
| --- | ---: |
| Pixtral vision tower | 218 |
| Multimodal projector | 4 |
| Ministral3 language model | 308 |
| `language_model.score.weight` | 1 |
| **Total** | **531** |

The real recipe is parsed without flattening or rewriting: nested
`text_config.rope_parameters` (`rope_type=yarn`, theta 1e6, factor 16,
original 16384, beta_fast 32, beta_slow 1, mscale 1/1,
`llama_4_scaling_beta` 0.1), nested `text_config.llama_4_scaling`
(`beta=0.1`, original 16384), `is_causal=false` global bidirectional attention,
262144 positions, `multimodal_projector_bias=false`, Pixtral 1024/24 layers/
16 heads/d64/patch14/merge2/1540 and feature layer `-1` resolved to the tower
output.

## Implemented

- `mistral3.rs`: composite config, exact header inventory, media policy, image
  packing, 2-D rotary row/column positions, spatial-merge source rows, marker
  validation, cold upload plan, and typed admission reasons.
- `mistral3_exec.rs`: owned Pixtral tower + projector + fusion. Uploads every
  vision/projector matrix through the bounded `WeightUploader`, binds the
  existing FA4 provider, and enqueues patch convolution, RMS, Q/K/V, 2-D
  split-half rotary, packed per-image attention, residual/add, gated SiLU,
  feature-layer selection, projector norm, spatial merge and the two-layer
  projector. One retained arena; no per-request allocation/query/sync.
- `Mistral3Engine`: joins the owned tower, the shared text decoder, pooled/raw
  readout and the optional Native score head (`language_model.score.weight`
  read once as BF16 and multiplied in FP32 over the mean-pooled rows). Score
  width comes from the checkpoint shape (`labels x hidden`), not a hard-coded
  C1 width. Production `submit_embed`/`submit_predict` stage into a
  `HostIoLease`, enqueue all device work (including fusion), record one
  completion event and return `Inference::Deferred`; only `Deferred` waits and
  packs. Two per-request works may be in flight, with independent drop and
  out-of-order completion; a failed fence quarantines its work and marks the
  engine unhealthy. `Mistral3Qualification::open` remains the synchronous
  qualification entry point.
- Shared decoder additions: `DecoderFamily::Mistral`, `attention_window`, YaRN
  tables, `llama_4_scaling` query scales, bidirectional global `gqa4`, native
  model-dtype mean pooling for Mistral3, and full-token readout. Two distinct math recipes:
  - `MistralH4096Ordinary` is the `parity_catalog` plain-Mistral contract
    (`infinity_rms_norm_4096_vector`, `infinity_add_rms_norm_4096_vector`,
    `infinity_silu_mul_bf16_vector8_native`).
  - `Mistral3H4096Transformers` is this wrapper's Native `_transformers_math`
    recipe (`infinity_rms_norm_rounded`, `infinity_add_rms_norm_rounded`,
    activation code 0 `infinity_silu_mul_f32`), selected only when the composite
    config sets `transformers_math`.
- Device source additions (source-only until Root AOT): 2-D rotary, add,
  scatter, pixel cast, merge gather, SiLU-mul, GELU, H128 base/scaled rotary
  tables, llama_4 query scale, the candidate Mistral3 model-dtype mean
  (`BF16 sum -> BF16 divide -> F32`, ATen reduction order unqualified), and a gated GELU/SiLU tower activation.
  Existing canonical26 FA4
  profiles are reused: vision = noncausal global `bert` (16/16 d64), text =
  noncausal global `gqa4` (32/8 d128), plain causal local = `mistral_local4096`.
- `native/`: b66 CAPI runner, Rust consumer, CPU metadata control, and
  `root_paired_controller.py` for Root paired Native-gold vs fresh Oxide
  candidate vectors, repeatability, lifecycle and profile-window performance.
  The controller refuses to run without `CUDA_VISIBLE_DEVICES` and
  `INFINITY_OXIDE_ROOT_GPU0_LOCKED=1`, and hash-checks the original config/index
  before and after the run.

## Native support boundary

Native Mistral3 supports text + image and a sequence-classification score head.
Native rejects audio and video for this processor. This packet therefore admits
image and text only and keeps video/audio unsupported by construction; it does
not claim video support.


## Structural geometry and factory hook

- Official wrapper geometries are structural, not a single H4096 checkpoint:
  3B H3072/I9216/L26, 8B H4096/I14336/L34 and 14B H5120/I16384/L40, all
  Q32/KV8/d128 with the same YaRN/llama4 conventions and Pixtral H1024 tower.
- `Mistral3Transformers` covers any Mistral3 wrapper hidden width with shaped
  base/YaRN tables and llama4 query scales.
- Plain causal Mistral H4096 uses the upstream `MistralH4096Ordinary` contract;
  plain causal H3072 uses `MistralH3072Exact` (Candle 1x4 norm + HFMA rotary);
  other ordinary widths use `MistralOrdinaryGeneric` (generic fused norm/NeoX).
- Flat `model_type=mistral` configs with `layer_types` and
  `sliding_window=32768` are admitted. Native reads the scalar window uniformly
  and ignores `layer_types`; an inactive window is normalized to the global
  mask. The 32768 tail is covered by the H4096 ordinary recipe.
- `params.json` short-key official configs (14B) are normalized in memory.
- `OxideBackend::new` now reaches Mistral3 through a cold hook
  (`CheckpointArchitecture::Mistral3`, `LoadedConfig::Mistral3`,
  `ExecutableModel::Mistral3`) instead of being qualification-only. The CPU
  test proves the hook reaches device admission (`no CUDA-capable device`) and
  does not fall through to encoder validation.
- FA4 absolute heads are runtime-dynamic: selection uses grouping ratio,
  head_dim, dtype and mask class. No new FA4 profile is required for the new
  geometries. `native/root_device_abi.json` lists the additive device exports.

## Unified host consumer

`native/host-consumer` is a standalone binary that uses the normal
`OxideBackend::new` factory (not `Mistral3Qualification`) for text embedding,
image embedding, classifier score, deferred lease and backend lifetime. It is a
binary only and is not part of any optimized all-test-target build. The granted
single optimized consumer build uses `opt-level=3`, `lto=false`,
`debug-assertions=false`, `codegen-units=16`, offline, CPU 136-143, four jobs,
empty `CUDA_VISIBLE_DEVICES`, and a unique task-owned target dir.

## Native SDK CPU stage

`native/root_native_cpu_stage.py` runs the prebuilt b66 C++ wrapper on CPU
before any GPU stage. It verifies the original config/index hashes and 531
tensor headers, runs the existing frozen SDK under the qualified owned
`remote_guard.py` pattern, loads all 4 shards, and emits per-case vectors. It
does not recompile the Native implementation, import Torch in the author
process, or create a CUDA context.

## Service metadata and weight-prefix contract

`core/src/model_metadata.rs` is the shared normal service helper: it reads
`config.json`, or normalizes `params.json` in memory when config.json is absent.
`download_artifacts` falls back to `params.json`, and the router uses the same
helper before `ModelConfig`/pooling/classifier resolution. This is source-wired
but the router crate cannot be compiled in the CPU-only author environment
because its LibTorch backend requires `LIBTORCH`; the core helper itself is
CPU-tested. Do not treat this as a runtime service claim until Root builds it.

The weight-prefix detector is explicit. HF-converted keys
(`model.language_model.*`, `model.vision_tower.*`, `language_model.score.weight`)
load normally. Raw Mistral-common keys (`tok_embeddings.*`, `layers.*`,
`vision_encoder.*`) return a typed error: Native and Oxide require an explicit
HF conversion, and this packet does not claim the raw Mistral-common format.

## Quality fixture positions

The Native CPU stage and host consumer include short-token cases with explicit
positions around 16384, 32768 and 262143, plus a shifted image case
(`image-yarn-16384`). This exercises YaRN and `llama_4_scaling` even when the
token count is small, instead of inert 0..7-only positions. The Native CPU stage
verifies the same vector names.

## Window semantics

For Mistral, a positive `sliding_window` is applied uniformly to every layer even
when it reaches the position limit. Flat 2410 `sliding_window=32768` is a local
mask (`left=32767, right=0`), not silently rewritten to global. `layer_types` is
not parsed by Native and is not invented per-layer here.

## Pooling, raw and arena policy

Mistral3 accepts mean, CLS and last-token pooling; CLS follows the Native
generic pool-0 path and is no longer rejected by the host. SPLADE/sparse output
is not claimed. API `raw_indices` selects whole sequences, not scattered tokens.
Validation rejects out-of-vocabulary ids, duplicate/out-of-range selectors,
zero/overflowing merge grids, oversized native image extents, wrong patch
payloads, and marker blocks that do not fit `nh*(nw+1)`. A vision checkpoint
with optional `.bias` tensors is rejected with a typed pre-load error because
the prepared vision linear path is bias-free; it is never silently ignored.
Mistral3 mean pooling uses a candidate Native model-dtype reduction (masked BF16 sum,
one BF16 round, BF16 divide, then widen F32) at every pooled-row count; the
shared non-Mistral3 decoder keeps V6 BF16 tree <=65535 and F32 fallback above.
Oversized active GPU arenas above the 256 MiB cache budget remain supported
uncached; the composer does not reject them globally or raise that budget. Only
retained and host-staging ownership stays bounded. Score output width is
`language_model.score.weight.shape[0]`, so a cached C2 checkpoint yields C2
values while the actual cached C1 checkpoint remains the main gate.

## Central device-owner request

`DEVICE-REQUEST.md` asks `/root/parity_catalog` to add the two new source
exports (`infinity_mistral3_model_dtype_mean_bf16` and
`infinity_mistral3_gelu_mul_bf16`) to the next additive device union (likely
131; union 130 is currently held). No device/AOT compile, GPU, Torch, Docker or
network run was made in this packet.

## Staged controller

`native/root_staged_controller.py` is the executable host controller for the
existing SDK container. It stages only the runner/header/scripts allowlist,
builds the runner against the frozen b66 SDK inside the container, runs the
inner owned `remote_guard.py`, validates the actual 531-tensor `/bench`
checkpoint and SDK/GOT 130200 before/after with `CUDA_VISIBLE_DEVICES=''`, and
only later runs the same controller on GPU0 for actual CAPI quality/perf. It
does not copy weights, start a container, or modify the SDK/weights.
## CPU evidence

- oxide lib debug/default: **653 passed, 0 failed, 72 ignored** (no all-targets run; compile-time policy)
- core lib (media + metadata): **28 passed**; shared metadata: **2 passed**
- Mistral3 config/engine filter: **15 passed**
- shaped/YaRN rotary filter: **4 passed**
- core media: **28 passed**
- cold Mistral3 factory hook: **1 passed**
- clippy `-D warnings` and rustfmt: clean
- CPU admission CLI: actual cached checkpoint, `vision_tensors=218`,
  `projector_tensors=4`, `text_tensors=308`, `score_tensors=1`,
  `classifier=true`, `text_pending=none`, `projector_pending=none`
- Native b66 CPU control: 531 tensors / 4 shards / 218 / 4 / 308 / 1,
  531 shape checks, payload unread, SDK `b6607c...`
- Paired-controller offline preflight: pass; config/index hashes match

## Root-pending (not claimed)

- AOT compilation of the additive operators and the FA4 profiles for
  sm_90/sm_100/sm_120; the device crate is source-only here.
- Real GPU execution and paired Native-gold / fresh-candidate comparison,
  including trained image/text/score vectors, lifecycle and perf.
- Whole-checkpoint quality, safety and performance claims.
- Actual 3B H3072, 14B H5120 and older 2410 trained checkpoints are structurally
  admitted and CPU-tested, but their GPU gold/lifecycle/perf remains Root-only.
- Native graph defaults remain intact; Oxide still uses no CUDA graphs.
