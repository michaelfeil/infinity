# Nemotron 3 and 3.5 embeddings

Use the LibTorch backend for these NVIDIA checkpoints:

| Checkpoint | Inputs | Dimensions |
| --- | --- | --- |
| `nvidia/Nemotron-3-Embed-1B-BF16` | Text | 2048 |
| `nvidia/Nemotron-3-Embed-8B-BF16` | Text | 4096 |
| `nvidia/Nemotron-3.5-Embed-1B-BF16` | Text, document images | 2048 |
| `nvidia/Nemotron-3.5-Embed-8B-BF16` | Text, document images | 4096 |

The saved Sentence Transformers configuration selects mean pooling and a
32768-token input limit. Attention is bidirectional. The native backend respects
independent attention head dimensions, YaRN, and position-dependent query scaling.
Pooling includes prefix and image-marker tokens, sums and divides in model dtype,
then converts to FP32 for normalization. Embedding endpoints return unit vectors
by default. `/embed_all` returns unpooled token states.

```sh
infinity_v3 --backend libtorch \
  --model-id nvidia/Nemotron-3.5-Embed-1B-BF16 --dtype bfloat16
```

Select the checkpoint's retrieval prefix on `/embed`:

```json
{"inputs": "Where are the forests?", "prompt_name": "query"}
```

Use `prompt_name: "document"` for passages. These names resolve to `query: ` and
`passage: `, respectively. There is no automatic prefix when neither a request
prompt nor a server default is selected. `--default-prompt-name document` selects
a server default; a request's `prompt_name` takes precedence.

For 3.5, named prefixes are rendered as a system message through the checkpoint's
retrieval template. That template omits instruction-model BOS/EOS wrappers,
escapes control markers in user text, and places images before text. Explicit
messages work on `/v1/embeddings`, `/embed`, and `/embed_all`:

```json
{
  "input": [
    {"role": "system", "content": "passage: "},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "A document page."}
    ]}
  ]
}
```

For `/embed` and `/embed_all`, use `inputs` instead of `input`. A named/default
prefix accepts one user message or a string; combining it with an explicit
system prefix is rejected. Without a named/default prefix, the native template
accepts one user message with an optional leading system prefix. Images use the
existing host allowlist, download and memory limits. Audio/video are unsupported.

The 3.5 models retain the frozen Nemotron 3 text encoder. Ordinary text with the
same prefix shares its embedding space; 3.5's template additionally escapes user
control markers. Source weights retain their [OpenMDW-1.1 license](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16).

Nemotron 3 text models and the frozen text encoder inside the 3.5 checkpoints
accept `--enable-fp8-dynamic true` on supported CUDA GPUs; the 3.5 vision tower
stays BF16. FP8 keeps the native quantized arithmetic path, so on 3.5 it bypasses
the Transformers-compatible rounding used by the unquantized path. Expect lower
cosine agreement with the BF16 reference and verify accuracy on your task before
enabling it. The reference parity checks below use unquantized weights.

## Verification

The parity runner uses Transformers 5.12.1 and a local checkpoint, comparing
queries, documents, control-marker escaping, images, multiple images, normalization,
and packed text batches through HTTP:

```sh
python scripts/nemotron-embed-parity.py \
  --checkpoint /path/to/Nemotron-3.5-Embed-1B-BF16 \
  --binary target/debug/infinity_v3 --output /tmp/nemotron-check \
  --dtype bfloat16 --device cuda:0 --extract-text
```

Run separately for each downloaded checkpoint to qualify its exact revision.
`--tiny` creates deterministic small weights with the released architecture,
including the 1B model's independent attention width. `--extract-text` additionally
saves and checks the frozen text encoder as a standalone Ministral3 model; this
does not independently verify a downloaded Nemotron 3 snapshot.

With `--reference-only`, the same runner writes fixtures for direct native and
frontend tests without an HTTP listener:

```sh
MISTRAL3_REFERENCE=/tmp/nemotron-check/native \
MISTRAL3_MODEL=/path/to/Nemotron-3.5-Embed-1B-BF16 \
MISTRAL3_TEST_POOLING=mean MISTRAL3_TEST_DTYPE=bfloat16 \
  cargo test -p infinity-backend-libtorch --test mistral3_parity -- --ignored --nocapture

NEMOTRON_REFERENCE=/tmp/nemotron-check \
NEMOTRON_MODEL=/path/to/Nemotron-3.5-Embed-1B-BF16 \
  cargo test -p infinity_v3-core --lib nemotron_preparation_matches_transformers -- --ignored
```

To test the extracted encoder directly, use `native-text` for the reference and
`text-model` for the model. Native tests compare every token coordinate and pooled
outputs in a packed batch. Frontend tests compare token IDs, image patch values,
and named/default prefix handling. FP32 uses strict absolute tolerances; BF16
also checks cosine and relative error to allow attention-backend rounding.
