# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

from typing import Any, Optional, Sequence

__all__ = ["truncate_texts_to_tokens"]


def truncate_texts_to_tokens(
    tokenizer: Any,
    texts: Sequence[str],
    max_tokens: Sequence[Optional[int]],
) -> list[str]:
    """Head-truncate each text to its first ``max_tokens[i]`` tokens.

    Used for both the query and each document before the (query, document) pair is built.
    A text is returned unchanged when its cap is ``None``/non-positive or it already fits,
    so text that does not need shortening never makes a lossy decode round-trip.

    Run this from the single preprocessing thread (i.e. inside ``encode_pre``) with that
    model's own ``tokenizer``; the token-counting ``_infinity_tokenizer`` runs on a
    different thread and must not be shared.
    """
    positive_caps = [n for n in max_tokens if n and n > 0]
    if not positive_caps:
        return list(texts)

    # Bound the tokenisation cost: a capped text only needs ``cap + 1`` tokens to detect
    # that it overflows, and an uncapped text is returned unchanged (its ids are discarded).
    # Truncating the batch to the largest cap + 1 avoids fully tokenising oversized inputs,
    # which is the OOM the limits exist to prevent.
    token_ids = tokenizer(
        list(texts),
        add_special_tokens=False,
        truncation=True,
        max_length=max(positive_caps) + 1,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    truncated: list[str] = []
    for original, ids, cap in zip(texts, token_ids, max_tokens):
        if cap and cap > 0 and len(ids) > cap:
            truncated.append(tokenizer.decode(ids[:cap], skip_special_tokens=True))
        else:
            truncated.append(original)
    return truncated
