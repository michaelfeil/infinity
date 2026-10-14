# System One input design

Status: typed wire schema implemented; native message inference awaits a model adapter.

Keep `POST /v1/systemone` in the existing infinity_v3 HTTP server, with the existing
question types and batching infrastructure. The canonical input forms are:

## Plain text

```json
{
  "state": "The customer requests a refund.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is a refund requested?"}
  }
}
```

## Native messages

```json
{
  "state": {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "The delivery arrived like this."},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
  },
  "questions": {
    "damage": {"type": "noul", "instructions": "Is the package damaged?"}
  }
}
```

This uses the explicit native-chat state envelope from
[openjev-sglang](https://github.com/ekzhang/openjev-sglang). Our contract accepts
only plain text or this envelope; bare arrays and arbitrary objects are rejected.
There is no second top-level messages field or separate media collection.

The shared Rust contract in `infinity_v3_core::input` is
`ModelInput::Text(String)` or `ModelInput::Messages(MessageInput)`.
`SystemOneInput` remains a compatibility alias. Message content is either a string or
ordered `ContentPart` values: Text, ImageUrl, InputAudio, VideoUrl. Initial roles
are system, developer, user, assistant. Tool messages require a future explicit
schema for call identifiers and metadata; they are currently rejected.

## Shared embedding API

The embedding endpoints use `EmbeddingInput`, sharing `Message`, `MessageRole`,
`MessageContent`, and `ContentPart` with the decision API. The outer envelope
remains endpoint-specific:

| Input | Meaning |
| --- | --- |
| `"hello"` | One text input |
| `["hello", "world"]` | Two independent text inputs |
| `[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]` | One conversation input |

Use `inputs` for `/embed`, `/embed_sparse`, and `/embed_all`; use `input` for
`/v1/embeddings`. `/embed` and `/v1/embeddings` return one embedding per conversation,
not one per turn. `/embed_all` retains its per-token output format. A batch of
conversations, mixed string/message arrays, arbitrary objects, and numeric token-ID
arrays are rejected. An empty array is an empty text batch and returns 400.
Removing numeric token-ID arrays is a breaking HTTP embedding API change, including
on `/v1/embeddings`; clients must send text. `/decode` retains its token-ID input.
The decision API continues to use `state: {"messages": [...]}`.

**Current capability:** text-only user/assistant conversations use the checkpoint's
native chat template through Basetenkenizer. Template compilation happens once at
startup and rendering runs in the bounded tokenizer workers, including when token
encoding falls back to Hugging Face. Ordered text parts are concatenated within
each message. Message order and roles are preserved. System/developer roles,
images, audio, and video return 422 before rendering, so templates cannot silently
drop unsupported content.

The server loads `chat_template.jinja` or the single/default `chat_template` in
`tokenizer_config.json`, using the model revision. A missing/invalid template
disables conversation processing while leaving ordinary text available; ambiguous
named templates require a `default` entry. Conversations use
`add_generation_prompt=false` and `add_special_tokens=false`: they embed the
supplied turns without starting an assistant reply or duplicating template BOS/EOS.
Checkpoint-specific generation-prefix policies require a model processor.

The native template owns conversation instructions. The plain-text default prompt
is not prepended to conversations, and explicit `prompt_name` with messages is
rejected. Over-limit conversation character counts are rejected before rendering;
rendered character limits also reject rather than cutting template markers. Token
truncation still follows the existing explicit/default truncation configuration.
Plain strings and string batches retain their existing prompt and tokenization
behavior. This change performs no image inference or remote media downloads.

## Semantics

- Parse messages into typed roles and ordered content blocks. Preserve their
  structure until the model adapter applies its native template and processor.
- Each question independently evaluates the same context. The adapter defines
  question placement according to the checkpoint's training/reference format.
- Text and image/audio/video blocks remain in their original order. Unsupported
  roles, content types, or input modes return validation errors; never silently
  stringify or discard native message content.
- Keep the existing question and answer contract. Model-specific action scores
  are optional; probability calibration belongs to the adapter.
- The HTTP contract uses familiar chat content blocks: text, image_url,
  input_audio, and (as an explicit extension) video_url. Actual support is
  capability-dependent, not implied by accepting the request schema.
- Resolve messages and media before queueing model-ready inputs. Reuse the infinity_v3
  queue, admission, cancellation, and replicas. Carry media alignment and resource
  costs with prepared inputs; token-only accounting is insufficient for media.
- Decode shared media once per request where possible. Do not promise shared
  transformer computation across questions.

## Compatibility and implementation boundary

This deliberately narrows the earlier PR's arbitrary JSON state contract. Clients
sending records must serialize them explicitly as text. Native message envelopes
are recognized structurally and never fall back to JSON text. The Laya adapter
returns 422 for native messages because it has no native chat processor or
multimodal encoder. Existing plain-text tokenization and output remain unchanged.

Jev-Omni's reference uses one user turn per question. First validate exact
single-turn processing and scores, then evaluate multi-turn question placement.
An ordinary Qwen checkpoint additionally needs a defined decision scoring method.
Neither model support nor multimodal Candle execution follows from a schema change.

Sources:

- https://docs.typesafe.ai/api
- https://github.com/ekzhang/openjev-sglang
- https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal
- https://huggingface.co/akhilaaa3/Jev-Omni/blob/main/jev_omni.py
- https://docs.vllm.ai/en/latest/features/multimodal_inputs/
