# Qwen3.5 / Qwen3.5-MoE vision contract (grounded)

Ground truth read on this host:

- Native model source: `/workspace/model-performance/michaelfeil/onejev-perf/mfu-next/source/backends/libtorch/cpp/models/qwen35_models.cpp`
  and `models/qwen3_vl_models.cpp`.
- Native vision kernels: `kernels/qwen35_vision.cu`, `kernels/qwen35_kernels.cu`.
- Native batching: `helpers/vision_batch.h`.
- Original cached checkpoints under
  `/workspace/model-performance/michaelfeil/onejev-perf/{backbone-comparison-20261023,models}`.
- Preprocessors: each full snapshot's `preprocessor_config.json` and
  `video_preprocessor_config.json`.

## Root and nested aliases

`create_qwen35` accepts exactly `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`,
`qwen3_5_moe_text`. Published conditional-generation roots use:

| checkpoint | root `model_type` | `text_config.model_type` | `vision_config.model_type` | architecture |
|---|---|---|---|---|
| 0.8B / 2B / 4B | `qwen3_5` | `qwen3_5_text` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` |
| 35B-A3B | `qwen3_5_moe` | `qwen3_5_moe_text` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` |

There is no separate `qwen35_vl` model type. The root itself contains
`vision_config`, and the shared Qwen3-VL tower is constructed only when media is
actually needed. `vision_config.model_type` is not the HF tower name; Native only
gates tower construction on the root `model_type`/`vision_config.depth` and on a
media-bearing request. Text-only roots
(`Qwen3_5ForCausalLM`, `Qwen3_5TextModel`, `Qwen3_5Model`, MoE equivalents) are
rejected by this VL leaf instead of being admitted without a tower.

## Vision configuration actually consumed

`hidden_size`, `num_heads`, `spatial_merge_size`, `depth`, `intermediate_size`,
`patch_size`, `temporal_patch_size`, `in_channels`,
`num_position_embeddings`, `out_hidden_size`, `hidden_act`,
`deepstack_visual_indexes`. Published values: patch 16, temporal patch 2, merge
2, 2304 positions (48x48), `out_hidden_size == text_config.hidden_size`, empty
deepstack list for every published root.

## Native tower versus Qwen3-VL

Qwen3.5 reuses the Qwen3-VL tower implementation (`Qwen3VL` with
`only_vision=true`). The actual differences are:

1. **Patch projection.** Native `exact_vision` is true only for CUDA FP16 and a
   root `model_type` starting with `qwen3_5`. In that arm the Conv3D kernel
   `[H,C,2,16,16]` is split into two compact `[H,768]` matrices, two GEMMs run,
   their BF16 sum is rounded, then one bias is added. For the published BF16
   checkpoints Native takes the non-exact arm: one flattened
   `[H, C*2*16*16]` GEMM with a fused bias epilogue. This leaf implements both,
   selected by dtype exactly as Native does.
2. **Deepstack.** Published Qwen3.5 roots have `deepstack_visual_indexes: []`,
   so no deepstack merger tensors exist and no deepstack residuals enter text.
   The code still validates and can execute a non-empty list on the host, and
   refuses a non-empty list on the device path rather than silently ignoring it.
3. **Text RoPE selectors.** Qwen3.5 uses `rope_parameters` with
   `partial_rotary_factor=0.25`, so head_dim 256 rotates only 64 values.
   `mrope_section=[11,11,10]` sums to `rotary/2 = 32`, not `head_dim/2 = 128`.
   Qwen3-VL uses `rope_scaling` and full-width mRoPE.
4. **Cross-modal GDN positions.** Only full-attention layers call `rotate`.
   Gated-delta/linear-attention layers never consume multimodal positions; their
   recurrence state is position-free. The fusion plan classifies every layer and
   must not project media positions into GDN state.

## Positions, rotary, interpolation

- Learned positions: square `pos_embed.weight` (`side = round(sqrt(2304)) = 48`);
  rows are indexed by `row*side + column`; `row/column` are merge-major patch
  coordinates scaled by `(side-1)/(grid-1)`; four corners are interpolated and
  summed in model dtype.
- Vision rotary: split-half 2-D rotary with
  `freq[j] = 1 / 10000^(2j/(head_dim/2))`; row phases occupy the first half of
  the rotary half-width, column phases the second half.
- Attention is non-causal and isolated per temporal frame (`cu_seqlens` over
  `height*width` frame rows). FA4 is used on device; there is no FA2 and no
  CUDA-graph path in Oxide.
- Merger grouping is merge-group-major. Final merger normalizes per hidden row
  then views `[merged, hidden*merge^2]`; deepstack mergers normalize the grouped
  row.

## Tiled media and video

- Image processor: `Qwen2VLImageProcessorFast`, shortest edge 65536 px,
  longest edge 16777216 px.
- Video processor: `Qwen3VLVideoProcessor`, shortest edge 4096 px, longest edge
  25165824 px.
- `smart_resize` uses `factor = patch_size * merge_size = 32`, round/floor/ceil
  exactly as the pinned processor.
- `grid_thw = [temporal_patches, height/16, width/16]`; tokens are
  `t*h*w/(merge^2)`. Video frames are grouped in temporal pairs; an odd frame
  count is rejected rather than silently duplicated.
- Packed pixel rows are `[C, T, P, P]`, ordered merge-group-major.
- External text positions advance by `max(merged_height, merged_width,
  temporal_patches)` after each media span, matching the Qwen2VL/Qwen3VL
  position-id remap.

## Cached checkpoint presence (metadata only)

| checkpoint | tensors | `model.visual.*` | depth | text hidden | preprocessors |
|---|---:|---:|---:|---:|---|
| Qwen3.5-0.8B | 488 | 153 | 12 | 1024 | yes |
| Qwen3.5-2B | 632 | 297 | 24 | 2048 | yes |
| Qwen3.5-4B | 738 | 297 | 24 | 2560 | yes |
| Qwen3.5-35B-A3B | 1811 | 333 | 27 (H1152, 16 heads, d72, I4304) | 2048 | config only in this cached copy |
| Elephant-Qwen3.5-VL | 760 | 333 | 27 (H1152/16/d72) | 4096 | combined processor_config |
| Mammoth-Qwen3.5-VL | 1184 | 333 | 27 (H1152/16/d72) | 5120 | combined processor_config |

The 35B weight snapshot carries the original config, index and 14 shards but not
the two preprocessor JSON files; the source/config admission and inventory test
uses the original config without inventing media policy.

## Implemented in this leaf

- Root alias/config admission, media preprocessor parsing, smart resize,
  merge-major patch mapping, video temporal pairing, batching, media spans.
- Real host tower forward over the actual original tensors, including BF16 and
  FP16-exact patch projection arms, position interpolation, 2-D rotary,
  per-frame attention, gated MLP, merger and deepstack host path.
- Real additive device exports and a device tower executor over the same generic
  typed `Bf16`/`LtProvider`/FA4 primitives. No FA2, no CUDA graph.
- External partial mRoPE plan, media position remap, GDN/full-attention layer
  classification, and exact visual-row replacement of text embeddings.
- CPU tests and consumer CLI; Root paired controller and Native tower probe.

## Not implemented / not claimed

- The owning `Qwen35VlEngine` now attaches to the public text backend, copies
  merged visual rows into the gathered embedding slab, and injects per-request
  external partial-mRoPE tables for full-attention layers. Pooled/raw and causal
  Decision submission share the same engine completion/retirement path.
- Elephant/Mammoth full checkpoints pass the text owner's full candidate gate.
  The 35B MoE text engine remains with its owning MoE worker; this leaf supplies
  the shared vision/fusion hook for that owner.
- No GPU run, device compile, Docker, Torch CUDA, NVCC, network or performance
  claim was made by the author.
- No public route, factory admission, model admission, or quality claim.


## Dtype coverage and Native weight exceptions (explicit)

Device coverage is **BF16 only**:
- text engine: BF16 weights/activations;
- vision device: `LtPlan<Bf16>`, FA4 BF16, and BF16-only `qwen35vl` pointwise
  kernels;
- the CPU reference recipe is also BF16-rounded. The F16 patch-split branch in
  `qwen35_vl_vision.rs` is a source-structure path only; it is not an F16 device
  or F16 reference recipe. F16/F32 device engines are not attached.

Native `bridge.cpp:379` preserves two GDN decay parameters in F32 rather than
casting the checkpoint value to the model dtype: `linear_attn.A_log` and
`linear_attn.dt_bias`. The shared foundation now reads them through
`WeightSource::read_f32`, so a stored-F32 value is not round-tripped through
BF16. Stored BF16/F16 values widen exactly. Every other F32 weight, including
ordinary `norm.weight`, is converted to the BF16 target exactly as Native does.

Actual 4B metadata: 738 tensors = **690 BF16 + 48 F32**, where the F32 set is
24 `linear_attn.A_log` + 24 `linear_attn.norm.weight` (dt_bias is stored BF16 in
that checkpoint). Elephant/Mammoth are all-BF16. No weight file was altered.


## Normal factory routes and dynamic-head FA4 modes

The published FA4 ABI bakes the Q/KV **ratio**, `head_dim`, dtype, causal bit
and local/global family. Absolute Q and KV counts are runtime DLTensor shapes;
Rust continues to validate the runtime q/kv shapes and element strides at launch.
The cold run-plan matcher therefore uses exact ratio + head_dim + causal + global,
not absolute heads.

| checkpoint | family | text mode | vision mode | factory |
|---|---|---|---|---|
| Elephant | dense | ratio4 d256 noncausal 6 | ratio1 d64/d72 global 8 | ready |
| Mammoth | dense | ratio6 d256 noncausal 16 | ratio1 d72 global 8 | ready |
| Qwen3.5-0.8B | dense | Q8/KV2 ratio4 d256 causal 18 | Q12/KV12 ratio1 d64 global 0 | ready |
| Qwen3.5-2B | dense | Q8/KV2 ratio4 d256 causal 18 | Q16/KV16 ratio1 d64 global 0 | ready |
| Qwen3.5-4B | dense | Q16/KV4 ratio4 d256 causal 18 | Q16/KV16 ratio1 d64 global 0 | ready |
| Qwen3.5-35B-A3B | MoE | Q16/KV2 ratio8 d256 causal new tail 26 | ratio1 d72 global 8 | MoE owner + new tail |

Only the 35B ratio8 global tail is new. Vision Q12/Q16 d64 reuse mode 0; text
Q8/KV2 and Q16/KV4 reuse mode 18; Mammoth reuses 17/16. No local or
different-ratio profile is borrowed. Requests: `device-packet/FA4-REQUEST.json`.

Noncausal runtime evidence: none of 0.8B, 2B, 4B or 35B declares
`use_bidirectional_attention` or a noncausal decision mode, so only their actual
causal text mask is required. Elephant/Mammoth declare `block_bidirectional` and
use their noncausal modes. If a future checkpoint declares bidirectional, the
plan includes that mask.

## Ordinary server route hook (no manual attachment)

The media hook is now inside the normal backend, not a user-side attach call:

1. `OxideBackend::new` -> `new_qwen35_serving` constructs the text route and, for
   a checkpoint with `vision_config`, validates the exact cold run plan and
   stores `Qwen35VlRuntime` in `OxideBackend`.
2. `Backend::embed`/`submit_embed` reads `Batch.multimodal`, converts each
   `ImagePatches` entry (grid, merge, per-sequence token offset) into a
   `Qwen35VlRequest`, builds the fused visual rows and external mRoPE table, and
   passes the fusion into the existing engine enqueue.
3. `Backend::decide`/`submit_warm_decision` performs the same conversion and
   keeps the fusion alive in the deferred completion closure.
4. A media Batch arriving at a backend without a VL runtime is rejected; images
   are never silently ignored. Audio is rejected.

`Qwen35VlEngine::from_normal_factory` remains an optional convenience wrapper;
the ordinary server route no longer requires it. `project`/`score` are the
pooled/readout and Decision aliases over the same owning engine; there is no
split text stand-in.

The serving tower path is device-resident: merged rows stay in a retained
`VisionDeviceOutput` workspace, copied row-to-row into the text embedding slab.
There is no visual-feature D2H/H2D round trip and no successful-path stream
synchronize. A `Qwen35VlInference` RAII owner is constructed before the deferred
closure is returned; if the closure is dropped uncalled it drives the inner
lease/fence to completion. An uncertain wait quarantines the fusion buffers
instead of freeing unproven GPU storage. The workspace is marked completed only
after the outer HostIo lease waits.

FP16 remains explicitly unsupported on device until the Native exact recipe is
implemented; no F16/F32 device claim is made.
