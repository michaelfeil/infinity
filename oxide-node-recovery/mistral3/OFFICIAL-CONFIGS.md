# Native-accepted Mistral/Ministral/Minstrall3 config matrix

| Model | Source | Text geometry | Text recipe | Vision | Notes |
| --- | --- | --- | --- | --- | --- |
| Ministral-3-3B-Instruct-2512-BF16 | official HF `config.json` | H3072, I9216, L26, Q32/KV8/d128, theta 1e6, eps 1e-5, ctx 262144, tied embeddings | nested `text_config.model_type=ministral3`; YaRN factor 16/orig 16384/beta 32,1/mscale 1,1; `llama_4_scaling_beta=0.1`; bidirectional wrapper (`_transformers_math`) | Pixtral H1024/I4096/L24/Q16/d64, patch14, merge2, SiLU, GELU projector, bias false, image1540 | local cache absent; official config verified by Root |
| Ministral-3-14B-Instruct-2512-BF16 | official HF `params.json` | dim 5120, I16384, L40, Q32/KV8/d128, theta 1e9, eps 1e-5, ctx 262144 | same YaRN/llama4 conventions; nested Mistral3 wrapper | identical Pixtral H1024 | `params.json` short-key schema normalized in memory; no checkpoint rewrite |
| Ministral-3-8B-Instruct-2512-BF16 | official HF `config.json` | H4096, I14336, L34, Q32/KV8/d128, theta 1e6, ctx 262144 | same YaRN/llama4 conventions; bidirectional wrapper | identical Pixtral H1024 | structural test added |
| Nemotron-3.5-Rerank-8B-BF16 | local cached actual | H4096, I14336, L34, Q32/KV8/d128, theta 1e6, ctx 262144, bidirectional | nested `ministral3` + YaRN + `llama_4_scaling`; classifier score head | Pixtral H1024/L24/Q16/d64/patch14/merge2 | 531 tensors, 4 shards; full CPU admission |
| Ministral-8B-Instruct-2410 | official HF `config.json` | flat `model_type=mistral`, H4096, I12288, L36, Q32/KV8/d128, theta 1e8, ctx 32768 | plain causal Mistral; scalar `sliding_window=32768`; `layer_types` present | none | Native reads `sliding_window` uniformly and ignores `layer_types`; the positive window is a local mask (`left=32767, right=0`), not silently global. The 32768 tail is admitted through the H4096 ordinary recipe. |
| Mistral-7B-v0.1 | local cached | H4096, I14336, L32, Q32/KV8/d128, theta 1e4, ctx 32768 | plain causal Mistral, `sliding_window=4096` | none | upstream plain-Mistral contract (`MistralH4096Ordinary`) |

## Required official fields if a config is fetched later

For each Mistral3 wrapper config, Root must provide the original file (never a
flattened rewrite) with:

- top-level: `model_type`, `architectures`, `text_config`, `vision_config`,
  `image_token_index`, `spatial_merge_size`, `vision_feature_layer`,
  `multimodal_projector_bias`, `projector_hidden_act`
- `text_config`: `model_type` (`ministral3` or `mistral`), `hidden_size`,
  `intermediate_size`, `num_hidden_layers`, `num_attention_heads`,
  `num_key_value_heads`, `head_dim`, `vocab_size`, `max_position_embeddings`,
  `rms_norm_eps`, `rope_theta`, `rope_parameters` (nested YaRN fields),
  `llama_4_scaling`, `hidden_act`, `is_causal`/`use_bidirectional_attention`,
  `sliding_window`
- `vision_config`: `model_type=pixtral`, `hidden_size`, `intermediate_size`,
  `num_hidden_layers`, `num_attention_heads`, `head_dim`, `patch_size`,
  `image_size`, `num_channels`, `hidden_act`, `rope_parameters`

For the 14B `params.json`, the short keys `dim`, `hidden_dim`, `n_layers`,
`n_heads`, `n_kv_heads`, `head_dim`, `vocab_size`, `norm_eps`, `rope_theta`,
`rope_parameters`/`yarn`, `llama_4_scaling`, `vision_encoder` are normalized in
memory.

## Geometry/recipe coverage

- Mistral3 wrapper (any hidden): `Mistral3Transformers` = Native
  `_transformers_math` rounded RMS/add-RMS, code-0 SiLU, shaped base/YaRN
  tables and llama4 query scales. Covers 3B H3072, 8B H4096 and 14B H5120.
- Plain causal Mistral H4096: `MistralH4096Ordinary` (upstream contract).
- Plain causal Mistral H3072: `MistralH3072Exact` (Candle 1x4 norm + HFMA
  rotary + code-5 SiLU).
- Other ordinary causal Mistral widths: `MistralOrdinaryGeneric` (generic
  fused norm/add-RMS, NeoX rotary, code-5 SiLU). Exactness beyond the registered
  Native widths is a Root GPU/gold qualification item, not a hidden config drop.
- Bidirectional plain Mistral without the wrapper: generic ordinary global path;
  bidirectional active local windows are rejected because Native's noncausal
  left-only window has no matching FA4 local form.
