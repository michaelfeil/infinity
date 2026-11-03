# Baseten Tokenizer

High-performance, Rust-backed BPE tokenization for inference. The Python
package is named `basetenkenizer`.

## Install

```bash
pip install basetenkenizer
```

Loading by Hugging Face model ID downloads `tokenizer.json` on first use and
then uses the local Hugging Face cache. Private or gated models use the usual
`HF_TOKEN` environment variable.

## Encode text

```python
from basetenkenizer import Tokenizer

tokenizer = Tokenizer.from_model("deepseek-ai/DeepSeek-V3.2")
encoding = tokenizer.encode(
    "A very long prompt that is now much faster.",
    add_special_tokens=False,
)

print(encoding.ids)
print(tokenizer.decode(encoding.ids))
```

`Tokenizer.from_file("tokenizer.json")` loads a local tokenizer. Encoding
objects expose `ids`, `attention_mask`, `type_ids`, and
`special_tokens_mask`; selected fields can be moved into NumPy arrays with
`encoding.into_numpy(...)`.

## Gemma 4

Gemma 4's Hugging Face BPE pipeline supports its space replacement normalizer,
byte fallback, and decoder sequence through the existing loading API:

```python
from basetenkenizer import Tokenizer

tokenizer = Tokenizer.from_model("google/gemma-4-31B-it")
ids = tokenizer.encode("Hello, world!", add_special_tokens=False).ids
```

Run offline Hugging Face parity with `cargo test --test gemma`. An optional
full-vocabulary test uses `GEMMA4_TOKENIZER_JSON` and `-- --ignored`.

## RoBERTa

Single-text RoBERTa tokenization supports `<s>` / `</s>` insertion, `<mask>`
whitespace handling, and older JSON files without an explicit BPE model type.

```python
from basetenkenizer import Tokenizer

tokenizer = Tokenizer.from_model("FacebookAI/roberta-base")
ids = tokenizer.encode("Hello <mask>!", add_special_tokens=True).ids
```

Sentence pairs and character/word alignment are unsupported. The existing
`special_tokens_mask` output does not identify post-processor inserted tokens.
Run offline Hugging Face parity with `cargo test --test roberta`.

## Native tiktoken.model files

Load a local ranks file or a directory containing `tiktoken.model` and optional
`tokenizer_config.json` without installing the Python `tiktoken` package:

```python
from basetenkenizer import Tokenizer

tokenizer = Tokenizer.from_file("/models/Kimi-K3")
# For a custom ranks file, specify its exact pre-tokenization regex:
tokenizer = Tokenizer.from_tiktoken_model(
    "/models/custom.model", pattern=r"\S+|\s+",
    special_tokens={"<|end|>": 1000},
)
```

Kimi configurations automatically select the Kimi regex and reserved-token
window. Other configurations can declare `pat_str` or `pattern`. A raw file
without configuration uses the converter's existing default regex; specify
`pattern` when another tokenization scheme is intended. Identified unknown
families without a declared regex are rejected. `from_model` continues to load
Hugging Face `tokenizer.json`; native ranks loading uses local paths.

## Llama 2 and Llama 3

Llama 2's SentencePiece-style BPE pipeline is supported through Hugging Face
`tokenizer.json`, including byte fallback, normalization-aware added tokens,
and decoder stripping. Llama 3's ByteLevel pipeline has offline parity coverage.

```python
from basetenkenizer import Tokenizer

tokenizer = Tokenizer.from_model("NousResearch/Llama-2-7b-chat-hf")
ids = tokenizer.encode("Hello!", add_special_tokens=True).ids
```

Use ordinary `encode` for Llama prompts. `encode_segments` is specialized for
Kimi K3 and does not guarantee Llama tokenization parity across segment boundaries.

Run offline parity with `cargo test --test llama --test normalized_added_tokens`.
The Llama 2 fixture includes its upstream license and use policy and is excluded
from published packages. Python parity uses Tokenizers 0.22.2, matching Rust's
reference dependency; 0.23.2 differs in normalized special-token skipping.

## Kimi K2.7 Code

Load the published tokenizer directly from its model repository:

```python
from basetenkenizer import Tokenizer

tokenizer = Tokenizer.from_model("moonshotai/Kimi-K2.7-Code")
ids = tokenizer.encode("def hello():\n    return 'world'").ids
```

For chat prompts, use the complete official model renderer. The
`encode_segments` contract documented below is specialized for Kimi K3.

## Kimi K3

Kimi K3 uses XTML and a Python chat renderer rather than a Jinja chat
template. Raw text can still be encoded normally:

```python
from basetenkenizer import Tokenizer

tokenizer = Tokenizer.from_model("moonshotai/Kimi-K3")
ids = tokenizer.encode("Hello from Kimi K3").ids
```

For rendered chat, pass the official Kimi K3 renderer's `EncodeSegment`
objects to `encode_segments` as `(segment.text, segment.allow_special)`. A
minimal user message followed by an assistant-generation prefix looks like
this:

```python
user_message = "Explain speculative decoding."

segments = [
    ("<|open|>", True),
    ("message", False),
    (" role", False),
    ('="', False),
    ("user", False),
    ('"', False),
    ("<|sep|>", True),
    (user_message, False),
    ("<|close|>", True),
    ("message", False),
    ("<|sep|>", True),
    ("<|end_of_msg|>", True),
    ("<|open|>", True),
    ("message", False),
    (" role", False),
    ('="', False),
    ("assistant", False),
    ('"', False),
    ("<|sep|>", True),
    ("<|open|>", True),
    ("response", False),
    ("<|sep|>", True),
]

encoding = tokenizer.encode_segments(segments)
input_ids = encoding.ids
```

Use Kimi K3's official `build_chat_segments` renderer for production
conversations involving tools, images, reasoning content, response schemas,
or multiple message types. Do not concatenate its segments before encoding:
doing so loses the special-token safety boundary.

## Kimi K3 segment encoding contract

`encode_segments` is specialized for Kimi K3's segment-based renderer.
Pass the official renderer's segments in order, preserving their text and
`allow_special` flags. Each segment is normalized, pre-tokenized, and BPE-encoded
independently before token IDs are joined. Segment boundaries are part of this
contract: the result need not equal encoding the concatenated text.

This is not a general-purpose chat encoding API for other models. For example,
Llama 2's `Prepend` normalizer runs independently on each nonempty text span,
so segmenting a prompt can introduce extra prefix markers. Use ordinary
`encode` for other model pipelines unless their renderer explicitly requires
and validates these segment semantics.

Each segment is a `(text, allow_special)` pair:

- `allow_special=True` recognizes tokenizer control tokens emitted by a trusted
  chat renderer.
- `allow_special=False` treats control-token-looking strings in user, tool, or
  attribute content as ordinary text.

`tiktoken_safe=True` is the default. It reproduces the chunk boundaries used
by legacy tiktoken tokenizers, including on very long inputs, so token IDs stay
compatible. Set it to `False` only when exact tiktoken parity is not required
and whole-segment BPE encoding is intentional.

Post-processing, truncation, and padding are applied after all segment IDs
have been joined.

## Use with Transformers

Call `patch_transformers` before loading a tokenizer:

```python
import basetenkenizer

basetenkenizer.patch_transformers()

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
tokens = tokenizer("Hello, world!")
```

Transformers v4 and v5 are supported. Pass
`patch_transformers(apply_chat_template=True)` to also use the native renderer
for supported render-only `apply_chat_template(..., tokenize=False)` calls;
unsupported templates automatically fall back to Transformers.

Baseten Tokenizer is focused on inference and does not implement every Hugging
Face Tokenizers training or alignment feature.

## License

The `basetenkenizer` package is licensed under the
[Apache License 2.0](LICENSE).
Third-party and derived components remain subject to their respective terms;
see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the release audit.
Some files are vendored from other projects for testing, including files under
`vendored_tokenizers/`, and remain subject to their respective terms rather
than Apache-2.0. Some test and benchmark files contain or were bundled from
internal implementations; they carry separate terms in their file headers and
are not available under Apache-2.0. Non-Apache materials are excluded from
published source and binary packages.

[View `basetenkenizer` on PyPI](https://pypi.org/project/basetenkenizer/)

## Acknowledgements and upstream contributions

Basetenkenizer builds on work from
[Crusoe's fastokens](https://github.com/crusoecloud/fastokens),
[gigatoken](https://github.com/marcelroed/gigatoken), and
[MiniJinja](https://github.com/mitsuhiko/minijinja). We're proud to contribute
to the open-source community and grateful for the work that made this project
possible. In addition to these libraries, various other libraries are used as
optional or required dependencies.

While developing Basetenkenizer, we also contributed improvements upstream:

- [crusoecloud/fastokens#8: add CI for cross-compiled PyO3 builds and upgrade PyO3](https://github.com/crusoecloud/fastokens/pull/8)
- [crusoecloud/fastokens#9: add a Tokio and PyO3 async interface](https://github.com/crusoecloud/fastokens/pull/9)
- [mitsuhiko/minijinja#913: render `None` and booleans like Jinja2](https://github.com/mitsuhiko/minijinja/pull/913)
- [mitsuhiko/minijinja#912: reset loop-local variables between iterations](https://github.com/mitsuhiko/minijinja/pull/912)
- [huggingface/tokenizers#1968: add a PCRE2 backend and regex sharding for faster tokenization](https://github.com/huggingface/tokenizers/pull/1968)
