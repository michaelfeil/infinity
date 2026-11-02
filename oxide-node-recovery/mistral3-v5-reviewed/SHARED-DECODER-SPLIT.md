# Shared decoder split (Mistral3 v3)

Base: current `/tmp/infinity-plain-cuda` Root working tree at public `55380b7`
plus the 625DenseV5 + Qwen35 + MeanV6 six-file overlay. The composer is a source
delta on that current working tree; it does not apply the obsolete V4 upstream
patch and does not write/reset the Root worktree. V5 positive Llama window
rejection and V6 BF16 mean <=65535 / F32 fallback >65535 are preserved.

## Current-base files shared with V5/Qwen35/MeanV6

- `decoder_model.rs`: `DecoderFamily`, `DecoderConfig`, weight loading, decoder classifier bundle.
- `decoder.rs`: generic `DecoderMath` variants, `DecoderKernels`, `DecoderWork`, classifier seam.
- `decoder_classifier.rs`: decoder head contracts.
- `lib.rs`: decoder classifier factory path and standard `Engine`/`Workspace` plumbing.
- `native_rope_tables.rs`: `decoder_128_tables` for the plain-Mistral contract.
- `cublas.rs`: classifier GEMM support.
- `lib.rs`: Qwen35 runtime/factory variants and the shared Engine/Workspace.
- `native_rope_tables.rs`: Qwen35 tables plus the plain-Mistral contract.

## Deltas added by this packet

- `decoder_model.rs`: `YarnRope`, `Llama4QueryScale`, `attention_window` parse,
  `ministral3` alias acceptance, flat-Mistral `layer_types` acceptance, YaRN
  parsing with `llama_4_scaling` fallback, `transformers_math` flag.
- `decoder.rs`: `MistralH3072Exact`, `MistralOrdinaryGeneric`,
  `Mistral3Transformers`; shaped table selection; query-scale launch; visual
  row replacement (`DecoderVisual`, `enqueue_visual`); inactive-window
  normalization. `MistralH4096Ordinary` remains the upstream plain-Mistral
  contract and is not redefined.
- `native_rope_tables.rs`: shaped `decoder_tables(head_dim, ...)` and
  `decoder_scaled_tables(head_dim, ...)` plus YaRN/query-scale tables.
- `lib.rs`: `CheckpointArchitecture::Mistral3`, `LoadedConfig::Mistral3`,
  `ExecutableModel::Mistral3`, the cold `OxideBackend::new` hook and the
  Mistral3 embedding/classifier batch routes.
- New Mistral3 modules/media/device/native packet as listed in `MISTRAL3.md`.

## Attention-profile rule (FA4)

FA4 absolute head counts are runtime-dynamic; selection uses the GQA grouping
ratio, `head_dim`, dtype and causal/local mask class. Do not force exact
absolute heads. Examples:

- Pixtral 12/12 d64 noncausal global reuses the `bert` profile (index 0), as
  does 16/16 d64.
- Mistral Q32/KV8 d128 global reuses causal index 2 / noncausal index 13.
- Ministral3 H3072/H5120 Q32/KV8 d128 uses the same attention profile; only the
  norm/RoPE/math width recipe changes.

A window at or above `max_position_embeddings` is inactive and normalized to
the global mask, matching Native. Active causal windows use the local mask
class; absolute local extents are runtime scalars. Native reads the scalar
`sliding_window` and applies it uniformly to every layer; it does not parse
`layer_types`, so this packet preserves that actual behavior rather than
inventing per-layer schedules.


## MoE3/shared-next coordination

The full MoE3 + shared-next baseline is owned by
`/tmp/oxide-all-model-feature-matrix-v1/qwen3-moe-qwen35-dense-compose-v1/worktree`.
This composer does not touch `moe_*` files or that worktree. It applies to the
current six-file V5/Qwen35/MeanV6 overlay; if a MeanV6 successor changes those
files, re-generate the composer from the new `BASE-PINS.json` hashes.
