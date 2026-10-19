# Ministral 3 multimodal embeddings

The LibTorch backend supports the public
[`mistralai/Ministral-3-3B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)
checkpoint's Mistral3 wrapper, Ministral3 text backbone, and Pixtral vision tower.
It serves hidden-state embeddings through Infinity's embedding APIs. This is an
instruction checkpoint, not a trained embedding checkpoint; no autoregressive
chat-completion or document-generation endpoint is added.

Use a build/image containing LibTorch for this example; it is selected
automatically.

```sh
infinity_v3 \
  --model-id mistralai/Ministral-3-3B-Instruct-2512-BF16 \
  --dtype bfloat16 --pooling last-token \
  --max-batch-tokens 8192 --auto-truncate
```

Last-token pooling is the default for this family; mean pooling can be selected
explicitly. Attention follows the checkpoint's text configuration. Text and
image messages use the checkpoint's native chat template, including system and
turn markers. Prompt overrides are unsupported.

Send a conversation to `/v1/embeddings`:

```json
{
  "model": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
  "input": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "Describe this image."}
    ]}
  ],
  "encoding_format": "float"
}
```

`/embed` and `/embed_all` accept the same message list in `inputs`. Image URLs
follow the existing allowlist and memory-budget controls; inline images work
without a remote-host allowlist. Image detail must be `auto` or omitted. Audio
and video inputs are rejected. Images honor EXIF orientation before resizing.
Multimodal inputs exceeding the token budget are
rejected even when automatic text truncation is enabled. Image row/end markers
count toward the request's actual token budget, but only image-placeholder
positions receive projected vision features.

## Cross-encoder reranking

`Mistral3ForSequenceClassification` checkpoints such as
[`nvidia/Nemotron-3.5-Rerank-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16)
load as rerankers. The score is a bias-free linear over the masked mean of the final
hidden states, computed in FP32 after averaging in the model dtype, which matches the
checkpoint's `model.py`. Pooling flags do not apply.

```sh
infinity_v3 --backend libtorch --model-id nvidia/Nemotron-3.5-Rerank-8B-BF16 --dtype bfloat16
```

`/rerank` and pair inputs to `/predict` render each pair through the checkpoint's
chat template as a `query` message followed by a `document` message, so its escaping
and turn layout apply.

```json
{"query": "What is NVIDIA NeMo Framework?", "texts": ["NeMo Framework offers ..."], "raw_scores": true}
```

`/rerank` takes text documents. For image documents, send the pair to `/v1/classify` as
`query` and `document` messages, the request shape the checkpoint documents for vLLM.
Set `use_activation` to `false`: one-label classifiers otherwise return a softmax of
1.0, while `false` returns the raw relevance logit.

```json
{"messages": [
  {"role": "query", "content": [{"type": "text", "text": "What is NVIDIA NeMo Framework?"}]},
  {"role": "document", "content": [
    {"type": "text", "text": "NeMo Framework offers ..."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}],
 "use_activation": false}
```

## Reference validation

Reference fixtures use public Transformers 5.12.1 and public model artifacts:

```sh
python scripts/mistral3-reference.py --tiny --output /tmp/mistral3-reference
MISTRAL3_REFERENCE=/tmp/mistral3-reference \
  cargo test -p infinity-backend-libtorch --test mistral3_parity -- --ignored --nocapture
```

The fixture generator checks text, square/non-square images, upscaling,
longest-edge downscaling, multiple images and EXIF orientation (the reference
explicitly transposes EXIF before calling the pinned fast processor).
The integration test compares pooled embeddings and every hidden-state coordinate
in a packed batch with independent Transformers outputs. Omit `--tiny` and pass
`--device cuda` to generate reference outputs from the trained BF16 weights;
set `MISTRAL3_MODEL`, `MISTRAL3_TEST_DEVICE=cuda:0`, and
`MISTRAL3_TEST_DTYPE=bfloat16` for the corresponding native test.

Source adaptations retain Transformers' Apache-2.0 notices under
`backends/libtorch/cpp/vendor/transformers_pixtral/`.

The reference defaults to eager attention; `--attention sdpa` provides a second
Transformers baseline. BF16 attention backends can differ numerically on the
trained checkpoint, so validation reports cosine similarity and relative L2
error as well as maximum coordinate error. FP32 CPU comparisons provide the
stricter implementation check. No embedding retrieval-quality claim follows
from supporting this instruction checkpoint.

To verify HTTP preprocessing, token expansion and normalized output as well:

```sh
python scripts/mistral3-http-parity.py --fixtures /tmp/mistral3-reference \
  --binary target/debug/infinity_v3
```

For trained fixtures containing full-resolution downscaling, pass
`--max-batch-tokens 8192 --device cuda --dtype bfloat16` and `--model` with
the trained model path to the HTTP check.

For synthetic bidirectional attention with independent head dimensions and
saved mean pooling, generate fixtures with `--tiny --bidirectional --pooling mean`
and run the HTTP check with `--pooling auto`. Set `MISTRAL3_TEST_POOLING=mean`
for the packed native test with those fixtures. Mean pooling sums and divides in
the model dtype before FP32 normalization. Saved Sentence Transformers pooling
metadata takes precedence over the last-token default when no CLI override is set.
