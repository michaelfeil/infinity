# Mistral3/Pixtral processor fixtures

This directory contains a CPU-only fixture generated from the real cached
checkpoint metadata at:

`/workspace/model-performance/michaelfeil/benchmark-overview/models/nvidia--Nemotron-3.5-Rerank-8B-BF16`

## Real cached inputs

`make_fixtures.py` reads exactly three files and nothing else:

| file | bytes | sha256 |
|---|---:|---|
| `config.json` | 3074 | `e1f442a21c464cf54259e7ca2ab09f6777f50a61fbf017efed8e6044894b4303` |
| `processor_config.json` | 697 | `a06f54b1cdc97ba9f114b9fa304c554ebfd2385ddbc3ffe5247d8e3782642c34` |
| `tokenizer_config.json` | 21263 | `384291712b80958e8c039a8c8afc318a11b0a7d7231f082381050f9a0ea6cad0` |

Tensor payloads were not read. `tokenizer.json` was not read. No CUDA, torch,
network, or GPU device is used.

The following fixture fields are copied directly from those JSONs:

- `processor.patch_size = 14`
- `processor.spatial_merge_size = 2`
- `processor.image_mean = [0.48145466, 0.4578275, 0.40821073]`
- `processor.image_std = [0.26862954, 0.26130258, 0.27577711]`
- `processor.longest_edge = 1120`
- `processor.image_size = 1540`
- `processor.image_token_index = 10`
- `processor.vision_feature_layer = -1`
- `processor.projector_hidden_act = "gelu"`
- `processor.multimodal_projector_bias = false`
- `processor.text_config.rope_parameters` and `llama_4_scaling` exactly as cached
  (`yarn`, factor 16, original 16384, beta_fast 32, beta_slow 1, mscale 1,
  mscale_all_dim 1, llama_4_scaling_beta 0.1, theta 1000000)
- text geometry (`hidden_size`, `intermediate_size`, layers, heads, KV heads,
  `head_dim`, vocab, `max_position_embeddings`, `is_causal`, `sliding_window`)
- tokenizer special token strings (`bos`, `eos`, `unk`, `pad`) and
  `extra_special_tokens` names from `tokenizer_config.json`.

`tokenizer_config.json` does not contain numeric special token ids. The fixture
records that fact explicitly and separately records the ids available in
`config.json.text_config` (`bos=1`, `eos=2`, `pad=11`).

## Synthetic image pixels

The image pixels are **deterministic synthetic geometry**, not a real image
decode. No image file is read. For every patch row `row` and patch element
`col` in `0..587`:

```text
pixel[row, col] = ((row * 1009 + col * 1013) % 997) / 997.0
```

The two unique pixel assets (`grid_1x4x6_d588` and `grid_1x14x10_d588`) are
stored in `pixel_assets` with their F32-LE payload sha256. Image cases reference
those assets by id.

## Case inventory

`processor-cases.json` contains:

- `image_cases` (4): `dynamic_grid_1x4x6`, `dynamic_grid_1x14x10`,
  `duplicate_same_image_twice`, `mixed_sizes_ragged_batch`.
- `text_cases` (2): `ragged_cumulative_0_7_30_60`, `duplicate_sequences`.
- `readout_cases` (4): `pooled_mean_indices`, `raw_indices`,
  `mixed_pooled_and_raw`, `classifier_label_case`.
- `error_cases` (6): `wrong_patch_dim`,
  `grid_height_not_divisible_by_merge`, `token_span_crossing_sequence`,
  `marker_id_mismatch`, `overlapping_image_spans`, `zero_grid`.

For every image the fixture records `patch_dim = 3*14*14 = 588`,
`patch_rows = grid_h*grid_w`, `token_rows = (grid_h/2)*(grid_w/2)`,
`token_start`, `token_span`, `marker_indices` using

```text
token_start + floor(image_token / (grid_w/merge)) * (grid_w/merge + 1)
             + image_token % (grid_w/merge)
```

and `merge_source_rows` for every image token in
`[merge_row * merge + merge_col]` order (Mistral3 SpatialRowMajor).

The text ids are valid for `vocab_size = 131072` and include
`image_token_index = 10`.

## Regenerate

```bash
cd /tmp/oxide-all-model-feature-matrix-v1/mistral3-v2
CUDA_VISIBLE_DEVICES='' python3 fixtures/make_fixtures.py
```

The command writes `fixtures/processor-cases.json` and prints the case counts.
