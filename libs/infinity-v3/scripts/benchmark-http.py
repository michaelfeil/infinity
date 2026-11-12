#!/usr/bin/env python3
"""Reproducible cross-runtime HTTP benchmark client.

Measures an already-running Infinity, vLLM, or SGLang server. It does not launch
models or orchestrate servers. Numerical probes are smoke checks for
cross-runtime comparison, not task-accuracy evaluation.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import itertools
import json
import math
import os
import platform
import random
import socket
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import transformers
from transformers import AutoTokenizer

try:
    import aiohttp
except ModuleNotFoundError:  # pragma: no cover
    aiohttp = None
    import httpx2

SEED = 42
TIMEOUT_SECONDS = 120.0
POOL_SIZE = 256
UUID_PLACEHOLDER = "00000000-0000-0000-0000-000000000000"
UUID_IMAGE_STYLES = ("vllm_openai_image",)
_UUID_PROCESS_NONCE = uuid.uuid4().hex
_UUID_COUNTER = itertools.count(1)
_UUID_STATS = {"generated": 0, "unique": 0, "duplicates": 0,
               "first_samples": [], "last_samples": [],
               "digest": hashlib.sha256()}
_UUID_LAST_COUNTER = 0
WORDS = """the of and to in a is that for it as with was on be at by this from or an
are which but not have has had were been their its can will more one all when there who
so if out up into than them these then some could would should about after over such
through between under while where because many most other new first last long great
little own old right big high small large next early young important few public bad same
able different local social national real best better sure free full special easy clear
recent certain personal open red difficult available likely short single medical current
wrong private past foreign fine common poor natural significant similar hot dead central
happy serious ready simple left physical general environmental financial blue democratic
dark various entire close legal religious cold final main green nice huge popular
traditional cultural wide impossible actual potential additional successful practical
necessary economic strong deep heavy warm white beautiful obvious effective perfect
valuable interesting modern quiet bright clean safe soft solid strange useful weak
welcome western ancient complex dangerous distinct dramatic efficient extreme familiar
famous formal frequent healthy honest independent individual initial inner innocent keen
kind literary live logical loose loud lucky massive mature mean mental military minor
mixed moral narrow native near normal northern official ordinary original outside
particular patient peaceful permanent plain pleasant political positive possible precious
precise prepared present pretty previous primary prime probable professional proper pure
quick rapid rare raw regular relevant reliable remote responsible rich rough round royal
rural sacred sad scientific secure senior sensitive separate sharp silent silly silver
slight slim slow smart smooth sophisticated sore sour spare specific spiritual splendid
stable standard steady steep sticky stiff still straight strategic strict striking strong
structural stupid subsequent substantial subtle sudden sufficient suitable sunny super
superior surprising sweet swift symbolic systematic tall technical temporary tender
terrible thick thin thirsty thorough tight tiny tired total tough tremendous tricky
typical ugly ultimate unique universal unknown unusual upper urban urgent usual valid vast
verbal vertical vital vivid vocal voluntary vulnerable wealthy weird wet whole wild
willing wise wonderful wooden yellow""".split()

TASKS = ("embeddings", "rerank", "classify", "sparse", "token_embeddings",
         "token_classify", "decision", "image")


class ValidationError(Exception):
    pass


def _token_len(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def _pair_len(tokenizer, query, document):
    return len(tokenizer.encode(query, document, add_special_tokens=True,
                                truncation=False))


def _seed_int(*parts):
    raw = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _fresh_transport_uuid():
    """Return a fresh fixed-length UUID string from a process nonce + counter."""
    global _UUID_LAST_COUNTER
    counter = next(_UUID_COUNTER)
    if counter <= _UUID_LAST_COUNTER or counter >= (1 << 48):
        raise RuntimeError("transport UUID counter repeated or exhausted")
    _UUID_LAST_COUNTER = counter
    raw = f"{_UUID_PROCESS_NONCE[:20]}{counter:012x}"
    text = f"{raw[0:8]}-{raw[8:12]}-4{raw[13:16]}-a{raw[17:20]}-{raw[20:32]}"
    _UUID_STATS["generated"] += 1
    _UUID_STATS["unique"] += 1
    first = _UUID_STATS["first_samples"]
    if len(first) < 8:
        first.append(text)
    last = _UUID_STATS["last_samples"]
    last.append(text)
    del last[:-8]
    _UUID_STATS["digest"].update(text.encode())
    return text


def _uuid_proof():
    """Bounded, parent-verifiable proof of the per-request UUID policy."""
    return {
        "policy": "fresh UUID per vLLM image content part; process nonce + global "
                  "request counter; fixed 36-byte in-place patch of a precomputed "
                  "body; JSON encoding and image generation stay outside timed windows",
        "placeholder": UUID_PLACEHOLDER,
        "total_request_uuids": _UUID_STATS["generated"],
        "unique_request_uuids": _UUID_STATS["unique"],
        "duplicate_request_uuids": _UUID_STATS["duplicates"],
        "sample_first": list(_UUID_STATS["first_samples"]),
        "sample_last": list(_UUID_STATS["last_samples"]),
        "sha256_of_all_uuid_strings": _UUID_STATS["digest"].hexdigest(),
        "full_uuid_list_retained_in_output": False,
        "full_uuid_list_retained_in_memory": False,
        "uniqueness_check": "strictly increasing 48-bit counter within process nonce",
    }


def _cache_control():
    return {
        "transport_uuid": _uuid_proof(),
        "scope": "transport identifiers only; not a proof of inference cache behavior",
        "sglang_image_payload": {
            "style": "sglang_flat_text_image",
            "fields": ["text", "image"],
            "rejects_openai_role_content": True},
        "vllm_token_classify": {
            "endpoint": "/pooling", "request_task": "token_classify",
            "response": "data[0].data raw 2D score matrix + usage"},
        "vllm_sparse": {
            "style": "vllm_sparse", "response_format": "dense_vocab_splade",
            "nonzero_pairs_serialized": True, "minimum_full_vocabulary": 1001},
    }


def _materialize_body(item):
    """Patch one fresh UUID into a precomputed body; return (body, used_uuid)."""
    body = item["body"]
    placeholder = item.get("uuid_placeholder")
    if placeholder is None:
        return body, None
    token = _fresh_transport_uuid()
    if body.count(placeholder) != 1:
        raise RuntimeError("transport UUID placeholder must occur exactly once")
    return body.replace(placeholder, token.encode()), token


def _safe_words(tokenizer):
    lead = tokenizer.encode("the", add_special_tokens=False)
    safe = [word for word in sorted(set(WORDS))
            if len(tokenizer.encode("the " + word, add_special_tokens=False)) == len(lead) + 1]
    if not safe:
        raise RuntimeError("tokenizer has no append-safe single-token words")
    return safe


def _build_text(tokenizer, safe, starters, target, rng):
    pool = [word for length, word in starters if length <= target]
    if not pool:
        raise ValueError(f"target length {target} is below every starter word")
    starter = rng.choice(pool)
    count = target - _token_len(tokenizer, starter)
    for _ in range(8):
        text = starter + (" " + " ".join(rng.choices(safe, k=count)) if count else "")
        got = _token_len(tokenizer, text)
        if got == target:
            return text
        count = max(0, count + target - got)
    raise RuntimeError(f"could not build text with exactly {target} tokens")


def _build_pair(tokenizer, safe, starters, specials, target, rng):
    if target < 2 * specials + 2:
        raise ValueError(f"rerank length {target} is too small for a tokenized pair")
    query = _build_text(tokenizer, safe, starters, max(specials + 1, target // 4), rng)
    pool = [word for length, word in starters if length <= target]
    starter = rng.choice(pool)
    count = max(0, target - _pair_len(tokenizer, query, starter))
    for _ in range(12):
        document = starter + (" " + " ".join(rng.choices(safe, k=count)) if count else "")
        got = _pair_len(tokenizer, query, document)
        if got == target:
            return query, document
        count = max(0, count + target - got)
    raise RuntimeError(f"could not build rerank pair with exactly {target} tokens")


def _usage(body):
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return {}
    return {key: usage[key] for key in
            ("prompt_tokens", "completion_tokens", "total_tokens",
             "input_tokens", "output_tokens")
            if isinstance(usage.get(key), int) and usage[key] >= 0}


def _finite(values, label, allow_empty=False):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 and not allow_empty:
        raise ValidationError(f"{label} is empty")
    if array.size and not np.isfinite(array).all():
        raise ValidationError(f"{label} contains a non-finite value")
    return array


def _numbers(value):
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, list):
        return [number for item in value for number in _numbers(item)]
    return []


def _sparse_rows(body, expected_rows):
    if isinstance(body, list):
        if len(body) != expected_rows:
            raise ValidationError(f"sparse rows {len(body)} != {expected_rows}")
        return body[0]
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise ValidationError("sparse response has no known row list")
    if len(body["data"]) != expected_rows:
        raise ValidationError(f"sparse rows {len(body['data'])} != {expected_rows}")
    embedding = body["data"][0].get("embedding")
    if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
        embedding = embedding[0]
    if not isinstance(embedding, list):
        raise ValidationError("sparse row lacks token/value pairs")
    return embedding


def _dense_vocab_sparse(body, expected_rows):
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise ValidationError("dense-vocabulary sparse response lacks a data list")
    if len(body["data"]) != expected_rows:
        raise ValidationError(f"dense-vocabulary sparse rows {len(body['data'])} != {expected_rows}")
    embedding = body["data"][0].get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValidationError("dense-vocabulary sparse row lacks a flat embedding")
    if any(isinstance(value, (list, dict)) for value in embedding):
        raise ValidationError("dense-vocabulary sparse vector is not one flat row")
    values = _finite(embedding, "dense vocabulary SPLADE")
    if values.ndim != 1:
        raise ValidationError("dense vocabulary SPLADE is not one vector per input")
    if values.size <= 1000:
        raise ValidationError(
            f"dense vocabulary size {values.size} <= 1000; refusing to reinterpret "
            "an ordinary dense embedding as SPLADE")
    if (values < 0).any():
        raise ValidationError("dense vocabulary SPLADE contains a negative value")
    usage = _usage(body)
    if "prompt_tokens" not in usage:
        raise ValidationError("dense vocabulary SPLADE usage lacks prompt_tokens")
    pairs = [{"index": int(index), "value": float(value)}
             for index, value in enumerate(values) if value != 0.0]
    return {"format": "dense_vocab_splade", "values": pairs, "usage": usage,
            "dense_vocab_size": int(values.size), "nonzero_count": len(pairs)}


def validate_response(task, body, expected_rows=1, style=None):
    if task in ("embeddings", "image"):
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise ValidationError(f"{task} response lacks a data list")
        if len(body["data"]) != expected_rows:
            raise ValidationError(f"{task} rows {len(body['data'])} != {expected_rows}")
        row = body["data"][0]
        values = _finite(row.get("embedding"), "embedding")
        if values.ndim != 1:
            raise ValidationError("embedding is not one vector per input")
        usage = _usage(body)
        if "prompt_tokens" not in usage:
            raise ValidationError(f"{task} usage lacks prompt_tokens")
        return {"format": f"openai_{task}", "values": values, "usage": usage}
    if task == "token_embeddings":
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise ValidationError("token_embeddings response lacks a data list")
        if len(body["data"]) != expected_rows:
            raise ValidationError("token_embeddings row count mismatch")
        row = body["data"][0]
        matrix = row.get("embedding", row.get("data"))
        values = _finite(matrix, "token embedding matrix")
        if values.ndim != 2:
            raise ValidationError("token embedding output is not a matrix")
        usage = _usage(body)
        if "prompt_tokens" not in usage:
            raise ValidationError("token_embeddings usage lacks prompt_tokens")
        return {"format": "token_embedding_matrix", "values": values, "usage": usage}
    if task == "rerank":
        if isinstance(body, list):
            rows, fmt = body, "infinity_rerank_list"
        elif isinstance(body, dict) and isinstance(body.get("results"), list):
            rows, fmt = body["results"], "jina_rerank_results"
        else:
            raise ValidationError("rerank response has no known row list")
        if len(rows) != expected_rows:
            raise ValidationError(f"rerank rows {len(rows)} != {expected_rows}")
        scores, indices = [], []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("index"), int):
                raise ValidationError("rerank row lacks an integer index")
            scores.append(float(_finite(
                [row.get("score", row.get("relevance_score"))], "rerank score")[0]))
            indices.append(row["index"])
        if sorted(indices) != list(range(expected_rows)):
            raise ValidationError("rerank indices do not cover the expected rows")
        return {"format": fmt, "values": np.asarray(scores), "usage": _usage(body)}
    if task == "classify":
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise ValidationError("classify response lacks a data list")
        if len(body["data"]) != expected_rows:
            raise ValidationError("classify row count mismatch")
        row = body["data"][0]
        values = row.get("probs")
        if values is None:
            values = row.get("scores")
        if values is None and isinstance(row.get("score"), (int, float)):
            values = [row["score"]]
        array = _finite(values, "classify scores")
        if row.get("label") is not None and not isinstance(row["label"], str):
            raise ValidationError("classify label is not a string")
        usage = _usage(body)
        if "prompt_tokens" not in usage:
            raise ValidationError("classify usage lacks prompt_tokens")
        return {"format": "classify_scores", "values": array, "usage": usage}
    if task == "sparse":
        if style == "vllm_sparse":
            return _dense_vocab_sparse(body, expected_rows)
        rows = _sparse_rows(body, expected_rows)
        pairs = []
        for pair in rows:
            if not isinstance(pair, dict) or not isinstance(pair.get("index"), int):
                raise ValidationError("sparse value lacks an integer index")
            value = float(_finite([pair.get("value")], "sparse value")[0])
            if pair["index"] < 0:
                raise ValidationError("sparse index is negative")
            pairs.append({"index": pair["index"], "value": value})
        return {"format": "sparse_token_values", "values": pairs, "usage": _usage(body)}
    if task == "token_classify":
        if isinstance(body, dict) and isinstance(body.get("data"), list):
            if len(body["data"]) != expected_rows:
                raise ValidationError("token_classify row count mismatch")
            row = body["data"][0]
            values = _finite(row.get("data"), "token classification score matrix")
            if values.ndim != 2:
                raise ValidationError("token classification output is not a 2D score matrix")
            usage = _usage(body)
            if "prompt_tokens" not in usage:
                raise ValidationError("token_classify usage lacks prompt_tokens")
            return {"format": "vllm_token_classify_matrix", "values": values,
                    "usage": usage, "matrix_shape": list(values.shape)}
        if not isinstance(body, list) or len(body) != expected_rows or not isinstance(body[0], list):
            raise ValidationError("token_classify response is not one prediction list")
        nums = []
        for row in body[0]:
            if not isinstance(row, dict) or not isinstance(row.get("token_id"), int):
                raise ValidationError("token prediction lacks an integer token_id")
            results = row.get("results")
            if not isinstance(results, dict) or not results:
                raise ValidationError("token prediction lacks label scores")
            scores = _finite(list(results.values()), "token label scores")
            if scores.ndim != 1:
                raise ValidationError("token label scores are not one flat label map")
            nums.extend(scores.tolist())
        return {"format": "token_predictions", "values": _finite(nums, "token label scores", allow_empty=True), "usage": _usage(body)}
    if task == "decision":
        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise ValidationError("decision response lacks answers")
        answers = body["answers"]
        if len(answers) != expected_rows:
            raise ValidationError(f"decision answers {len(answers)} != {expected_rows}")
        nums = []
        for answer in answers.values():
            if not isinstance(answer, dict):
                raise ValidationError("decision answer is not an object")
            answer_nums = _numbers(answer)
            if not answer_nums:
                raise ValidationError("decision answer has no numeric field")
            nums.extend(answer_nums)
        usage = _usage(body)
        if "input_tokens" not in usage:
            raise ValidationError("decision usage lacks input_tokens")
        return {"format": "systemone_answers", "values": _finite(nums, "decision answer values"), "usage": usage}
    raise ValidationError(f"unsupported task {task}")


def _stats(values):
    return None if not values else {"count": len(values), "min": min(values),
                                    "max": max(values), "mean": sum(values) / len(values)}


def _workload_concurrency(task, length, configured):
    if task in ("embeddings", "rerank") and length >= 2048:
        return min(configured, 8)
    if task == "token_embeddings" and length >= 512:
        return min(configured, 8)
    if task == "image" and length == 1024:
        return min(configured, 8)
    return configured


class Workload:
    def __init__(self, tokenizer, safe, specials, task, length, seed=SEED, variant=None):
        self.tokenizer = tokenizer
        self.safe = safe
        self.specials = specials
        self.task = task
        self.length = length
        self.seed = seed
        self.variant = variant
        self.payload_style = None
        self.concurrency = 1
        self.starters = sorted((_token_len(tokenizer, word), word) for word in safe)
        self.images = self._make_images() if task == "image" else None
        self.items = []
        self.counter = 0

    def _make_images(self):
        from PIL import Image
        images = []
        size = self.length
        yy, xx = np.indices((size, size))
        for variant in range(4):
            array = np.stack(((xx + 17 * variant) % 256,
                              (yy + 29 * variant) % 256,
                              ((xx + yy) // 2 + 43 * variant) % 256), axis=-1).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(array, "RGB").save(buffer, format="PNG")
            images.append(buffer.getvalue())
        return images

    def raw_item(self, index):
        rng = random.Random(_seed_int(self.seed, self.task, self.length, index))
        if self.task == "rerank":
            query, document = _build_pair(
                self.tokenizer, self.safe, self.starters, self.specials, self.length, rng)
            item = {"query": query, "document": document,
                    "local_tokens": _pair_len(self.tokenizer, query, document)}
        elif self.task == "decision":
            state = " ".join(rng.choices(self.safe, k=self.length))
            count = self.variant if self.variant is not None else (1 if self.length <= 128 else (12 if self.length == 512 and index % 2 else 3))
            questions = {f"q{i}": {"type": "noul",
                                   "instructions": f"Is statement {i} true for this state?"}
                         for i in range(count)}
            item = {"state": state, "questions": questions, "state_words": self.length,
                    "local_tokens": _token_len(self.tokenizer, state)}
        elif self.task == "image":
            variant = index % len(self.images)
            image = self.images[variant]
            include_text = self.variant if self.variant is not None else bool(index % 2)
            text = f"Represent this image variant {index} for retrieval." if include_text else None
            content = [{"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(image).decode()}}]
            if text:
                content.append({"type": "text", "text": text})
            item = {"image_size": self.length, "image_sha256": hashlib.sha256(image).hexdigest(),
                    "image_variant": variant, "text": text, "input": [{"role": "user",
                    "content": content}], "local_tokens": _token_len(self.tokenizer, text)
                    if text else 0}
        else:
            text = _build_text(self.tokenizer, self.safe, self.starters, self.length, rng)
            item = {"text": text, "local_tokens": _token_len(self.tokenizer, text)}
        item["expected_rows"] = len(item["questions"]) if self.task == "decision" else 1
        item["sha256"] = hashlib.sha256(json.dumps(
            {key: value for key, value in item.items()
             if key not in ("local_tokens", "expected_rows")}, sort_keys=True).encode()).hexdigest()
        return item

    def prepare(self, style, model, size=POOL_SIZE):
        self.items = []
        self.payload_style = style
        for index in range(size):
            item = self.raw_item(index)
            body = _json_bytes(_payload(self.task, style, model, item))
            item["body"] = body
            item["body_sha256"] = hashlib.sha256(body).hexdigest()
            if style in UUID_IMAGE_STYLES:
                item["uuid_placeholder"] = UUID_PLACEHOLDER.encode()
            self.items.append(item)
        self.counter = 0
        return self

    def next(self):
        if not self.items:
            raise RuntimeError("workload pool was not prepared")
        item = self.items[self.counter % len(self.items)]
        self.counter += 1
        return item


def _payload(task, style, model, item):
    if task in ("embeddings", "token_embeddings"):
        if style == "vllm_pooling_token":
            return {"model": model, "input": [item["text"]], "task": "token_embed"}
        body = {"model": model, "input": [item["text"]], "encoding_format": "float"}
        if style.startswith("infinity"):
            body["truncate"] = False
        return body
    if task == "image":
        image_url = item["input"][0]["content"][0]["image_url"]["url"]
        if style == "sglang_image":
            # Installed SGLang 0.5.23 MultimodalEmbeddingInput accepts only flat
            # text/image/video fields, not OpenAI role/content chat parts.
            return {"model": model,
                    "input": [{"text": item["text"], "image": image_url}],
                    "encoding_format": "float"}
        content = [dict(part) for part in item["input"][0]["content"]]
        if style in UUID_IMAGE_STYLES:
            content[0] = {**content[0], "uuid": UUID_PLACEHOLDER}
        body = {"model": model,
                "input": [{"role": item["input"][0].get("role", "user"),
                           "content": content}],
                "encoding_format": "float"}
        if style.startswith("infinity"):
            body["truncate"] = False
        return body
    if task == "rerank":
        if style.startswith("infinity"):
            return {"model": model, "query": item["query"], "texts": [item["document"]],
                    "truncate": False, "raw_scores": True}
        payload = {"model": model, "query": item["query"], "documents": [item["document"]]}
        if style == "vllm_rerank":
            payload["use_activation"] = False
        return payload
    if task == "sparse":
        if style.startswith("infinity"):
            return {"model": model, "inputs": [item["text"]], "truncate": False}
        if style == "vllm_sparse":
            return {"model": model, "input": [item["text"]], "encoding_format": "float"}
        return {"model": model, "input": [item["text"]], "encoding_format": "float"}
    if task == "token_classify":
        if style.startswith("infinity"):
            return {"model": model, "inputs": [item["text"]], "truncate": False}
        if style == "vllm_token_classify":
            return {"model": model, "input": [item["text"]], "task": "token_classify"}
        return {"model": model, "input": [item["text"]]}
    if task == "decision":
        return {"model": model, "state": item["state"], "questions": item["questions"]}
    body = {"model": model, "input": [item["text"]]}
    if style.startswith("infinity"):
        body["truncate"] = False
    return body


def _json_bytes(payload):
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()


class WindowState:
    def __init__(self):
        self.attempted = self.responses = self.ok = 0
        self.latencies, self.local_tokens = [], []
        self.usage_prompt, self.usage_total, self.usage_input = [], [], []
        self.compared = self.equal = self.different = 0
        self.uuid_requests = 0
        self.concurrency = 1
        self.formats, self.failures, self.failure_count = set(), [], 0

    def failure(self, kind, detail, status=None, body=None):
        self.failure_count += 1
        if len(self.failures) < 20:
            self.failures.append({"kind": kind, "detail": str(detail)[:500], "status": status,
                                  "body": None if body is None else str(body)[:500]})


def finalize_window(state, start, end, target_tokens):
    elapsed = max(0.0, end - start)
    throughput = (state.ok / elapsed) if state.ok and elapsed > 0 else None
    return {
        "attempted": state.attempted, "responses": state.responses, "ok": state.ok,
        "errors": state.attempted - state.ok, "unanswered": state.attempted - state.responses,
        "elapsed_seconds": elapsed, "throughput_rps": throughput,
        "input_tokens_per_second_local": sum(state.local_tokens) / elapsed
        if state.ok and elapsed > 0 else None,
        "input_tokens_per_second_usage_input": sum(state.usage_input) / elapsed
        if state.ok and state.usage_input and elapsed > 0 else None,
        "latency_ms": {"p50": float(np.percentile(state.latencies, 50))
                       if state.latencies else None,
                       "p95": float(np.percentile(state.latencies, 95))
                       if state.latencies else None, "samples": len(state.latencies)},
        "tokens_per_request": {"local": _stats(state.local_tokens), "target": target_tokens,
                               "usage_prompt": _stats(state.usage_prompt),
                               "usage_total": _stats(state.usage_total),
                               "usage_input": _stats(state.usage_input)},
        "token_count_comparison": {"compared": state.compared, "equal": state.equal,
                                   "different": state.different,
                                   "usage_prompt_unavailable": state.ok - state.compared},
        "response_formats": sorted(state.formats),
        "uuid_requests": state.uuid_requests,
        "concurrency": state.concurrency,
        "failure_count": state.failure_count, "failure_samples": state.failures,
    }


def _probe_record(validated):
    values = validated["values"]
    if isinstance(values, list) and (not values or isinstance(values[0], dict)):
        flat = [entry["value"] for entry in values]
        shape = [len(values)]
    else:
        array = np.asarray(values, dtype=np.float64)
        values = array.tolist()
        shape = list(array.shape)
        flat = array.reshape(-1).tolist()
    record = {"response_format": validated["format"], "usage": validated["usage"],
              "shape": shape, "values": values, "all_finite": True,
              "norm": float(np.linalg.norm(np.asarray(flat, dtype=np.float64)))}
    for key in ("dense_vocab_size", "nonzero_count", "matrix_shape"):
        if key in validated:
            record[key] = validated[key]
    return record


class HttpClient:
    def __init__(self, base_url, concurrency):
        self.base_url, self.concurrency, self.session = base_url.rstrip("/"), concurrency, None

    async def __aenter__(self):
        if aiohttp is not None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS),
                connector=aiohttp.TCPConnector(limit=self.concurrency))
        else:
            self.session = httpx2.AsyncClient(
                timeout=httpx2.Timeout(TIMEOUT_SECONDS),
                limits=httpx2.Limits(max_connections=self.concurrency,
                                     max_keepalive_connections=self.concurrency),
                trust_env=False)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if aiohttp is not None:
            await self.session.close()
        else:
            await self.session.aclose()

    async def post_bytes(self, path, body):
        headers = {"content-type": "application/json"}
        if aiohttp is not None:
            async with self.session.post(self.base_url + path, data=body,
                                         headers=headers) as response:
                return response.status, await response.text(), dict(response.headers)
        response = await self.session.post(self.base_url + path, content=body, headers=headers)
        return response.status_code, response.text, dict(response.headers)

    @property
    def name(self):
        return "aiohttp" if aiohttp is not None else "httpx2"

    @property
    def version(self):
        return aiohttp.__version__ if aiohttp is not None else httpx2.__version__


def _candidates(runtime, task):
    if task == "image":
        if runtime == "infinity":
            return [("/v1/embeddings", "infinity_image")]
        if runtime == "vllm":
            return [("/v1/embeddings", "vllm_openai_image")]
        return [("/v1/embeddings", "sglang_image")]
    if task == "embeddings":
        return [("/v1/embeddings", "infinity_embeddings")] if runtime == "infinity" else \
            [("/v1/embeddings", "openai_embeddings")]
    if task == "rerank":
        if runtime == "infinity":
            return [("/rerank", "infinity_rerank"), ("/v1/rerank", "infinity_rerank")]
        if runtime == "vllm":
            return [("/rerank", "vllm_rerank"), ("/v1/rerank", "vllm_rerank")]
        return [("/v1/rerank", "jina_rerank"), ("/rerank", "jina_rerank")]
    if task == "classify":
        if runtime == "infinity":
            return [("/v1/classify", "infinity_classify")]
        if runtime == "vllm":
            return [("/classify", "openai_classify"), ("/v1/classify", "openai_classify")]
        return [("/v1/classify", "openai_classify"), ("/classify", "openai_classify")]
    if task == "sparse":
        if runtime == "infinity":
            return [("/embed_sparse", "infinity_sparse")]
        if runtime == "vllm":
            return [("/v1/embeddings", "vllm_sparse")]
        return [("/v1/embeddings", "openai_sparse")]
    if task == "token_embeddings":
        if runtime == "infinity":
            return [("/v1/embeddings", "infinity_embeddings")]
        if runtime == "vllm":
            return [("/pooling", "vllm_pooling_token"),
                    ("/v1/embeddings", "openai_embeddings")]
        return [("/v1/embeddings", "openai_embeddings")]
    if task == "token_classify":
        if runtime == "infinity":
            return [("/predict_tokens", "infinity_token_classify")]
        if runtime == "vllm":
            return [("/pooling", "vllm_token_classify")]
        return [("/token_classify", "openai_token_classify")]
    if task == "decision":
        return [("/v1/systemone", "systemone")]
    raise ValueError(f"unsupported task {task}")


async def _resolve_endpoint(client, runtime, task, workload, model):
    attempts = []
    for path, style in _candidates(runtime, task):
        item = workload.raw_item(-1)
        body = _json_bytes(_payload(task, style, model, item))
        body, _ = _materialize_body({
            "body": body,
            "uuid_placeholder": UUID_PLACEHOLDER.encode()
            if style in UUID_IMAGE_STYLES else None})
        try:
            status, text, headers = await client.post_bytes(path, body)
            if status != 200:
                raise ValidationError(f"HTTP {status}")
            validated = validate_response(task, json.loads(text),
                                          item["expected_rows"], style)
        except Exception as exc:
            attempts.append({"path": path, "style": style, "error": str(exc)[:500]})
            continue
        return {"path": path, "style": style, "status": status,
                "response_format": validated["format"],
                "transport_uuid_requests": 1 if style in UUID_IMAGE_STYLES else 0,
                "server_header": headers.get("server"),
                "content_type": headers.get("content-type"), "attempts": attempts}
    raise RuntimeError("no endpoint validated: " + json.dumps(attempts))


async def _attempt(client, workload, path, task, state, item=None):
    item = workload.next() if item is None else item
    state.attempted += 1
    start = time.monotonic()
    try:
        body, used_uuid = _materialize_body(item)
        if used_uuid:
            state.uuid_requests += 1
        status, text, _ = await client.post_bytes(path, body)
    except Exception as exc:
        state.failure("transport", repr(exc))
        return None
    finished = time.monotonic()
    state.responses += 1
    state.latencies.append((finished - start) * 1000.0)
    if status != 200:
        state.failure("http_status", f"HTTP {status}", status, text)
        return None
    try:
        validated = validate_response(task, json.loads(text), item["expected_rows"],
                                      getattr(workload, "payload_style", None))
    except (ValidationError, json.JSONDecodeError) as exc:
        state.failure("validation", exc, status, text)
        return None
    state.ok += 1
    state.local_tokens.append(item["local_tokens"])
    state.formats.add(validated["format"])
    usage = validated["usage"]
    if "prompt_tokens" in usage:
        state.usage_prompt.append(usage["prompt_tokens"])
        state.compared += 1
        state.equal += usage["prompt_tokens"] == item["local_tokens"]
        state.different += usage["prompt_tokens"] != item["local_tokens"]
    if "total_tokens" in usage:
        state.usage_total.append(usage["total_tokens"])
    if "input_tokens" in usage:
        state.usage_input.append(usage["input_tokens"])
    return validated


async def _run_closed_loop(client, workload, path, task, args, seconds, concurrency):
    state, start = WindowState(), time.monotonic()
    state.concurrency = concurrency
    deadline = start + seconds

    async def worker():
        while time.monotonic() < deadline:
            await _attempt(client, workload, path, task, state)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return finalize_window(state, start, time.monotonic(), workload.length)


async def _run_serial_probe(client, tokenizer, safe, specials, path, task, args, payload_style):
    length = 8 if task == "decision" else (256 if task == "image" else args.min_latency_length)
    workload = Workload(tokenizer, safe, specials, task, length).prepare(payload_style, args.model)
    state, start = WindowState(), time.monotonic()
    state.concurrency = 1
    for _ in range(20):
        await _attempt(client, workload, path, task, state)
    result = finalize_window(state, start, time.monotonic(), workload.length)
    result.update({"min_ms": min(state.latencies) if state.latencies else None,
                   "all_latencies_ms": state.latencies, "serial_requests": 20,
                   "length": length})
    return result


def _tokenizer_metadata(path):
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "added_tokens.json", "vocab.json", "merges.txt"):
        file = Path(path) / name
        if file.is_file():
            digest = hashlib.sha256()
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            files[name] = digest.hexdigest()
    return {"path": str(path), "files": files,
            "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def _client_metadata(args, client):
    return {"python": platform.python_version(), "platform": platform.platform(),
            "hostname": socket.gethostname(), "pid": os.getpid(),
            "http_client": client.name, "http_client_version": client.version,
            "numpy": np.__version__, "transformers": transformers.__version__,
            "tokenizer": _tokenizer_metadata(args.tokenizer)}


def _descriptor(workload, style):
    samples = []
    for item in workload.items[:3]:
        sample = {"sha256": item["sha256"], "body_sha256": item["body_sha256"],
                  "local_tokens": item["local_tokens"]}
        if workload.task == "rerank":
            sample.update(query=item["query"], document=item["document"])
        elif workload.task == "decision":
            sample.update(state=item["state"], questions=item["questions"],
                          state_words=item["state_words"])
        elif workload.task == "image":
            sample.update(image_size=item["image_size"], image_sha256=item["image_sha256"],
                          image_variant=item["image_variant"], text=item["text"])
        else:
            sample["text"] = item["text"]
        samples.append(sample)
    result = {"task": workload.task, "length": workload.length, "seed": workload.seed,
              "special_tokens": workload.specials, "safe_words": len(workload.safe),
              "payload_style": style, "pool_size": len(workload.items),
              "concurrency": workload.concurrency,
              "cycles_prepared_pool": len(workload.items) > 0, "samples": samples,
              "transport_uuid": {
                  "enabled": style in UUID_IMAGE_STYLES,
                  "placeholder": UUID_PLACEHOLDER if style in UUID_IMAGE_STYLES else None,
                  "placement": "vLLM image content part 'uuid' field" if style in UUID_IMAGE_STYLES else None,
                  "semantic_sha256_excludes_transport_uuid": True}}
    if workload.task == "image":
        result["image_variants"] = len(workload.images)
        result["includes_image_only_and_image_text"] = True
        result["image_payload_style"] = style
        result["image_payload_shape"] = (
            "sglang_flat_text_image" if style == "sglang_image"
            else "openai_chat_role_content")
        result["image_concurrency_policy"] = {
            "documented_h100_workers": {"256": 32, "1024": 8},
            "policy": "256px uses --concurrency (default 32); 1024px uses "
                      "min(--concurrency, 8); all runtimes paired; serial probe uses 1",
            "selected_workers": workload.concurrency,
            "paired_across_runtimes": True,
            "serial_latency_workers": 1,
        }
    return result


def _write_json_atomic(path, document):
    target = Path(path)
    os.makedirs(target.parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".benchmark-http-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _notes(client):
    notes = ["Numerical smoke only; not task-accuracy evaluation.",
             "All request bodies are precomputed before warm-up and timing; the prepared "
             "pool (>=256) is cycled, with no tokenization or serialization per timed request.",
             "Warm-up is time-based seconds; windows include drain and all validation in "
             "elapsed (end-start).",
             "Local token counts use AutoTokenizer(add_special_tokens=True, truncation=False); "
             "server usage is recorded separately and mismatches are not hidden.",
             "Rerank counts the tokenizer query/document pair; custom chat formatting can "
             "legitimately differ from server usage.",
             "Decision lengths are state words (8/128 use 1 question, 512 has separate 3/12-question cases, "
             "2048 uses 3); image sizes have separate image-only and image-plus-text cases.",
             "Image-only local token counts do not include model-specific image tokens; server "
             "usage is still recorded.",
             "SGLang image requests use its flat {text, image} embedding input, not OpenAI "
             "role/content chat parts; semantic image SHA and text stay cross-runtime comparable.",
             "vLLM image requests patch a fresh fixed-length content-part UUID into each "
             "precomputed body; JSON/image generation stays outside timed windows.",
             "vLLM token classification records the raw data[0].data 2D score matrix and usage; "
             "label/span alignment is not invented and mismatches are reported.",
             "vLLM SPLADE accepts a finite, nonnegative dense vocabulary vector only for the "
             "explicit vllm_sparse style; only nonzero index/value pairs are serialized.",
             "Image concurrency follows the H100 docs: 256px uses --concurrency (default 32), "
             "1024px uses min(--concurrency, 8), paired across runtimes; the serial probe "
             "remains single-worker.",
             "Token matrices use concurrency 32 at 128 tokens and 8 at 512 tokens to bound "
             "large JSON results within the servers' default response/staging budgets.",
             "The serial latency probe uses 8 tokens for text/token tasks, 8 state words for "
             "decision, and a 256px image for image.",
             "Caches are expected to be disabled by the orchestrator; cycling is disclosed."]
    if aiohttp is None:
        notes.append("aiohttp was unavailable; HTTP transport fell back to httpx2.")
    return notes


async def _benchmark(args, tokenizer, safe, specials):
    started = time.monotonic()
    failures, workloads = [], []
    runtime = {"requested": args.runtime, "url": args.url, "model": args.model,
               "task": args.task}
    async with HttpClient(args.url, args.concurrency) as client:
        try:
            endpoint = await _resolve_endpoint(
                client, args.runtime, args.task,
                Workload(tokenizer, safe, specials, args.task, args.lengths[0]), args.model)
            runtime.update(endpoint)
        except Exception as exc:
            endpoint = None
            failures.append({"phase": "endpoint_resolution", "detail": repr(exc)})
        if endpoint is None:
            return {"schema_version": 1, "kind": "cross_runtime_http_numerical_smoke",
                    "ok": False, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": time.monotonic() - started, "arguments": vars(args),
                    "runtime": runtime, "client": _client_metadata(args, client),
                    "workloads": workloads, "latency_probe": None, "failures": failures,
                    "cache_control": _cache_control(), "notes": _notes(client)}
        specs = [(length, None) for length in args.lengths]
        if args.task == "decision":
            specs = [(length, questions) for length in args.lengths
                     for questions in ([1] if length <= 128 else [3, 12] if length == 512 else [3])]
        elif args.task == "image":
            specs = [(length, text) for length in args.lengths for text in (False, True)]
        for length, variant in specs:
            try:
                workload = Workload(tokenizer, safe, specials, args.task, length, variant=variant)
                workload.concurrency = _workload_concurrency(
                    args.task, length, args.concurrency)
                workload.prepare(endpoint["style"], args.model)
                descriptor = _descriptor(workload, endpoint["style"])
                warmup = await _run_closed_loop(client, workload, endpoint["path"],
                                                args.task, args, args.warmup,
                                                workload.concurrency)
                windows = [await _run_closed_loop(client, workload, endpoint["path"],
                                                  args.task, args, args.seconds,
                                                  workload.concurrency)
                           for _ in range(args.repeats)]
                probe_item = workload.items[0]
                probe_state = WindowState()
                probe_valid = await _attempt(client, workload, endpoint["path"], args.task,
                                             probe_state, item=probe_item)
                probe_record = _probe_record(probe_valid) if probe_valid else None
                numerical_probe = {"sample": {key: value for key, value in probe_item.items()
                                              if key not in ("body", "input", "uuid_placeholder")},
                                   "record": probe_record,
                                   "failures": probe_state.failures,
                                   "failure_count": probe_state.failure_count}
                if (args.task == "token_classify" and probe_record
                        and probe_record.get("response_format") == "vllm_token_classify_matrix"):
                    rows = probe_record["shape"][0] if probe_record["shape"] else None
                    numerical_probe["alignment_diagnostic"] = {
                        "matrix_rows": rows,
                        "local_tokens": probe_item["local_tokens"],
                        "aligned": rows == probe_item["local_tokens"],
                        "note": "Raw vLLM token-classification matrix; no label/span "
                                "alignment is inferred."}
                workloads.append({"length": length, "variant": variant,
                                  "concurrency": workload.concurrency,
                                  "descriptor": descriptor,
                                  "warmup": warmup, "windows": windows,
                                  "numerical_probe": numerical_probe})
                for phase, summary in [("warmup", warmup)] + [
                        (f"window_{index}", window)
                        for index, window in enumerate(windows)]:
                    if summary["failure_count"]:
                        failures.append({"phase": phase, "length": length,
                                         "count": summary["failure_count"],
                                         "samples": summary["failure_samples"]})
                if probe_state.failure_count:
                    failures.append({"phase": "numerical_probe", "length": length,
                                     "count": probe_state.failure_count,
                                     "samples": probe_state.failures})
            except Exception as exc:
                failures.append({"phase": "workload", "length": length, "detail": repr(exc)})
        try:
            latency_probe = await _run_serial_probe(
                client, tokenizer, safe, specials, endpoint["path"], args.task, args,
                endpoint["style"])
            if latency_probe["failure_count"]:
                failures.append({"phase": "serial_latency_probe",
                                 "count": latency_probe["failure_count"],
                                 "samples": latency_probe["failure_samples"]})
        except Exception as exc:
            latency_probe = None
            failures.append({"phase": "serial_latency_probe", "detail": repr(exc)})
    return {"schema_version": 1, "kind": "cross_runtime_http_numerical_smoke",
            "ok": not failures, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - started, "arguments": vars(args),
            "runtime": runtime, "client": _client_metadata(args, client),
            "workloads": workloads, "latency_probe": latency_probe, "failures": failures,
            "cache_control": _cache_control(), "notes": _notes(client)}


def _load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


async def _mock_pipeline_test(tokenizer, safe, specials, tokenizer_path):
    if aiohttp is None:
        return "skipped: aiohttp unavailable"
    from aiohttp import web

    seen = {"vllm_image": [], "sglang_image": [], "vllm_sparse": [], "vllm_token": []}
    status = {"scenario": "vllm_image"}
    active = {"256": 0, "1024": 0}
    max_active = {"256": 0, "1024": 0}
    expected_by_length = {}
    for length in (256, 1024):
        workload = Workload(tokenizer, safe, specials, "image", length)
        for image in workload.images:
            url = "data:image/png;base64," + base64.b64encode(image).decode()
            expected_by_length[hashlib.sha256(url.encode()).hexdigest()] = length

    async def observe_image_concurrency(image_url):
        length = expected_by_length[hashlib.sha256(image_url.encode()).hexdigest()]
        key = str(length)
        active[key] += 1
        max_active[key] = max(max_active[key], active[key])
        try:
            await asyncio.sleep(0.02)
        finally:
            active[key] -= 1

    async def embeddings(request):
        scenario = status["scenario"]
        payload = await request.json()
        if scenario == "vllm_image":
            content = payload["input"][0]["content"]
            image_part = content[0]
            assert image_part["type"] == "image_url"
            token = image_part.get("uuid")
            assert isinstance(token, str) and len(token) == 36
            assert "uuid" not in payload["input"][0]
            image_url = image_part["image_url"]["url"]
            await observe_image_concurrency(image_url)
            seen["vllm_image"].append({
                "uuid": token,
                "image_sha256": hashlib.sha256(image_url.encode()).hexdigest(),
                "text": next((part["text"] for part in content
                              if part.get("type") == "text"), None)})
            body = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8}}
        elif scenario == "sglang_image":
            assert set(payload) == {"model", "input", "encoding_format"}
            part = payload["input"][0]
            assert set(part) == {"text", "image"}
            assert isinstance(part["image"], str)
            assert part["image"].startswith("data:image/png;base64,")
            await observe_image_concurrency(part["image"])
            seen["sglang_image"].append({
                "image_sha256": hashlib.sha256(part["image"].encode()).hexdigest(),
                "text": part["text"]})
            body = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8}}
        elif scenario == "vllm_sparse":
            assert isinstance(payload["input"], list)
            assert isinstance(payload["input"][0], str)
            vector = [0.0] * 1001
            vector[1], vector[1000] = 0.5, 1.5
            seen["vllm_sparse"].append(payload)
            body = {"data": [{"index": 0, "embedding": vector}],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8}}
        else:
            raise AssertionError(f"unexpected embeddings scenario {scenario}")
        return web.json_response(body)

    async def pooling(request):
        payload = await request.json()
        assert payload.get("task") == "token_classify"
        assert isinstance(payload.get("input"), list)
        assert isinstance(payload["input"][0], str)
        seen["vllm_token"].append(payload)
        return web.json_response({
            "data": [{"index": 0, "data": [[0.1, 0.2], [0.3, 0.4]]}],
            "usage": {"prompt_tokens": 8, "total_tokens": 8}})

    app = web.Application()
    app.router.add_post("/v1/embeddings", embeddings)
    app.router.add_post("/pooling", pooling)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    original_prepare = Workload.prepare

    def small_prepare(self, style, model, size=4):
        return original_prepare(self, style, model, size=4)

    Workload.prepare = small_prepare
    try:
        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"

        def make_args(task, runtime, lengths, seconds=0.05, warmup=0.02):
            return argparse.Namespace(
                url=url, runtime=runtime, model="benchmark", tokenizer=tokenizer_path,
                task=task, output="unused", lengths=lengths, concurrency=32,
                seconds=seconds, repeats=1, warmup=warmup, min_latency_length=8)

        def check_image_doc(doc):
            # Exercise the result-file boundary as well as live transport.
            json.dumps(doc, allow_nan=False)
            keys = set()
            for workload in doc["workloads"]:
                expected = 32 if workload["length"] == 256 else 8
                keys.add((workload["length"], workload["variant"]))
                assert workload["concurrency"] == expected
                descriptor = workload["descriptor"]
                assert descriptor["concurrency"] == expected
                assert descriptor["image_concurrency_policy"]["selected_workers"] == expected
                assert workload["warmup"]["concurrency"] == expected
                assert workload["warmup"]["attempted"] > 0
                assert workload["windows"] and all(
                    window["concurrency"] == expected and window["attempted"] > 0
                    for window in workload["windows"])
            assert keys == {(256, False), (256, True), (1024, False), (1024, True)}
            probe = doc["latency_probe"]
            assert probe and probe["concurrency"] == 1
            assert probe["serial_requests"] == 20

        def check_nonimage_doc(doc):
            workload = doc["workloads"][0]
            assert workload["concurrency"] == 32
            assert workload["descriptor"]["concurrency"] == 32
            assert workload["warmup"]["concurrency"] == 32
            assert all(window["concurrency"] == 32 for window in workload["windows"])
            assert doc["latency_probe"]["concurrency"] == 1

        def reset_active():
            active.update({"256": 0, "1024": 0})
            max_active.update({"256": 0, "1024": 0})

        reset_active()
        status["scenario"] = "vllm_image"
        vllm_image_doc = await _benchmark(
            make_args("image", "vllm", [256, 1024], seconds=0.15, warmup=0.05),
            tokenizer, safe, specials)
        assert vllm_image_doc["ok"], vllm_image_doc["failures"][:2]
        check_image_doc(vllm_image_doc)
        assert max_active == {"256": 32, "1024": 8}, max_active

        reset_active()
        status["scenario"] = "sglang_image"
        sglang_image_doc = await _benchmark(
            make_args("image", "sglang", [256, 1024], seconds=0.15, warmup=0.05),
            tokenizer, safe, specials)
        assert sglang_image_doc["ok"], sglang_image_doc["failures"][:2]
        check_image_doc(sglang_image_doc)
        assert max_active == {"256": 32, "1024": 8}, max_active

        for left, right in zip(vllm_image_doc["workloads"],
                               sglang_image_doc["workloads"]):
            for one, two in zip(left["descriptor"]["samples"],
                                right["descriptor"]["samples"]):
                assert one["sha256"] == two["sha256"]
                assert one["image_sha256"] == two["image_sha256"]
                assert one["text"] == two["text"]
        image_ids = [row["uuid"] for row in seen["vllm_image"]]
        assert image_ids and len(image_ids) == len(set(image_ids))
        proof = vllm_image_doc["cache_control"]["transport_uuid"]
        assert proof["total_request_uuids"] == proof["unique_request_uuids"]
        assert proof["duplicate_request_uuids"] == 0
        assert proof["total_request_uuids"] >= len(image_ids)

        assert seen["sglang_image"]
        assert {row["image_sha256"] for row in seen["sglang_image"]} <= set(
            expected_by_length)
        texts = {row["text"] for row in seen["sglang_image"]}
        assert None in texts and any(isinstance(text, str) for text in texts)
        assert {row["image_sha256"] for row in seen["vllm_image"]} <= set(
            expected_by_length)

        status["scenario"] = "vllm_token"
        token_doc = await _benchmark(make_args("token_classify", "vllm", [8]),
                                     tokenizer, safe, specials)
        assert token_doc["ok"], token_doc["failures"][:2]
        check_nonimage_doc(token_doc)
        token_window = token_doc["workloads"][0]["warmup"]
        assert "vllm_token_classify_matrix" in token_window["response_formats"]
        alignment = token_doc["workloads"][0]["numerical_probe"]["alignment_diagnostic"]
        assert alignment["matrix_rows"] == 2
        assert alignment["local_tokens"] == 8 and alignment["aligned"] is False
        assert seen["vllm_token"]

        status["scenario"] = "vllm_sparse"
        sparse_doc = await _benchmark(make_args("sparse", "vllm", [8]),
                                      tokenizer, safe, specials)
        assert sparse_doc["ok"], sparse_doc["failures"][:2]
        check_nonimage_doc(sparse_doc)
        sparse_window = sparse_doc["workloads"][0]["warmup"]
        assert "dense_vocab_splade" in sparse_window["response_formats"]
        sparse_probe = sparse_doc["workloads"][0]["numerical_probe"]["record"]
        assert sparse_probe["dense_vocab_size"] == 1001
        assert sparse_probe["nonzero_count"] == 2
        assert seen["vllm_sparse"]
    finally:
        Workload.prepare = original_prepare
        await runner.cleanup()
    return "mock pipelines ok"


def _self_test(args):
    pair = {"query": "query", "document": "document"}
    for runtime in ("infinity", "vllm", "sglang"):
        for _, style in _candidates(runtime, "rerank"):
            payload = _payload("rerank", style, "benchmark", pair)
            if runtime == "infinity":
                assert payload["raw_scores"] is True
            elif runtime == "vllm":
                assert payload["use_activation"] is False
            else:
                assert "use_activation" not in payload and "raw_scores" not in payload
    assert _workload_concurrency("embeddings", 2048, 32) == 8
    empty_ner = validate_response("token_classify", [[]], 1, "infinity_token_classify")
    assert empty_ner["values"].shape == (0,)
    for bad_score in (None, float("nan"), float("inf"), [0.5]):
        try:
            validate_response("token_classify", [[{"token_id": 1, "results": {"O": bad_score}}]], 1)
        except (ValidationError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid token-label score accepted")
    tokenizer = _load_tokenizer(args.tokenizer)
    safe, specials = _safe_words(tokenizer), _token_len(tokenizer, "")
    assert POOL_SIZE >= 256
    assert _workload_concurrency("image", 256, 32) == 32
    assert _workload_concurrency("image", 1024, 32) == 8
    assert _workload_concurrency("image", 1024, 4) == 4
    assert _workload_concurrency("image", 512, 32) == 32
    assert _workload_concurrency("embeddings", 1024, 32) == 32
    checks = [("embeddings", 8), ("sparse", 8), ("token_embeddings", 8),
              ("token_classify", 8), ("classify", 8), ("rerank", 128),
              ("decision", 512), ("image", 256)]
    for task, length in checks:
        workload = Workload(tokenizer, safe, specials, task, length)
        item = workload.raw_item(0)
        if task == "rerank":
            assert item["local_tokens"] == _pair_len(tokenizer, item["query"], item["document"])
            assert item["local_tokens"] == length
        elif task == "decision":
            assert len(item["state"].split()) == length
            assert item["expected_rows"] in (3, 12)
        elif task == "image":
            assert item["image_size"] == 256
            assert item["image_sha256"] and (not item["text"] or item["local_tokens"] > 0)
        else:
            assert item["local_tokens"] == length, (task, length, item["local_tokens"])
        workload.prepare("openai", "model", size=4)
        assert len(workload.items) == 4
        first = workload.next()
        workload.next()
        assert first["body"].startswith(b"{")
        assert workload.next() is not None

    image = Workload(tokenizer, safe, specials, "image", 32)
    image.prepare("vllm_openai_image", "model", size=4)
    image_item = image.items[0]
    image_payload = json.loads(image_item["body"])
    image_content = image_payload["input"][0]["content"]
    assert image_content[0]["uuid"] == UUID_PLACEHOLDER
    assert "uuid" not in image_payload["input"][0]
    assert image_item["body"].count(UUID_PLACEHOLDER.encode()) == 1
    template = image_item["body"]
    placeholder = image_item["uuid_placeholder"]
    tokens = set()
    for _ in range(1000):
        body, token = _materialize_body(image_item)
        assert token and len(token) == 36
        assert body.count(token.encode()) == 1
        assert body.replace(token.encode(), placeholder) == template
        tokens.add(token)
    assert len(tokens) == 1000

    sglang = Workload(tokenizer, safe, specials, "image", 32)
    sglang.prepare("sglang_image", "model", size=4)
    sglang_payload = json.loads(sglang.items[0]["body"])
    assert set(sglang_payload) == {"model", "input", "encoding_format"}
    sglang_part = sglang_payload["input"][0]
    assert set(sglang_part) == {"text", "image"}
    assert sglang_part["image"] == image_content[0]["image_url"]["url"]
    assert sglang_part["text"] is None and sglang_part["image"]
    sglang_with_text = json.loads(sglang.items[1]["body"])["input"][0]
    assert sglang_with_text["text"] == sglang.items[1]["text"]
    assert isinstance(sglang_with_text["text"], str) and sglang_with_text["image"]

    infinity = Workload(tokenizer, safe, specials, "image", 32)
    infinity.prepare("infinity_image", "model", size=2)
    infinity_payload = json.loads(infinity.items[0]["body"])
    assert "truncate" in infinity_payload and infinity_payload["truncate"] is False
    assert "uuid" not in infinity_payload["input"][0]["content"][0]

    sparse = Workload(tokenizer, safe, specials, "sparse", 8)
    sparse.prepare("vllm_sparse", "model", size=2)
    sparse_payload = json.loads(sparse.items[0]["body"])
    assert set(sparse_payload) == {"model", "input", "encoding_format"}
    assert sparse_payload["input"] == [sparse.items[0]["text"]]
    assert sparse_payload["encoding_format"] == "float"
    assert "uuid_placeholder" not in sparse.items[0]

    token_classify = Workload(tokenizer, safe, specials, "token_classify", 8)
    token_classify.prepare("vllm_token_classify", "model", size=2)
    token_payload = json.loads(token_classify.items[0]["body"])
    assert token_payload["task"] == "token_classify"
    assert isinstance(token_payload["input"], list) and isinstance(token_payload["input"][0], str)

    state = WindowState()
    state.attempted, state.responses, state.ok = 3, 3, 2
    state.latencies = [10.0, 20.0, 30.0]
    record = finalize_window(state, 0.0, 12.5, 128)
    assert math.isclose(record["elapsed_seconds"], 12.5), record
    assert math.isclose(record["throughput_rps"], 2 / 12.5), record
    assert record["errors"] == 1 and record["latency_ms"]["p50"] == 20.0
    good = {
        "embeddings": {"data": [{"index": 0, "embedding": [0.1, 0.2]}],
                       "usage": {"prompt_tokens": 8, "total_tokens": 8}},
        "image": {"data": [{"index": 0, "embedding": [0.1, 0.2]}],
                  "usage": {"prompt_tokens": 8, "total_tokens": 8}},
        "token_embeddings": {"data": [{"index": 0, "embedding": [[0.1, 0.2], [0.3, 0.4]]}],
                             "usage": {"prompt_tokens": 8, "total_tokens": 8}},
        "rerank": [{"index": 0, "score": 0.5}],
        "classify": {"data": [{"index": 0, "label": "A", "probs": [0.25, 0.75]}],
                     "usage": {"prompt_tokens": 8, "total_tokens": 8}},
        "sparse": [[{"index": 1, "value": 0.5}]],
        "token_classify": [[{"token": "a", "token_id": 1, "results": {"B": 0.5}}]],
        "decision": {"answers": {"q0": {"type": "noul", "noul": 0.5}},
                     "usage": {"input_tokens": 8, "output_tokens": 0}},
    }
    for task, body in good.items():
        validate_response(task, body, 1)
    try:
        validate_response("embeddings", {"data": [{"index": 0,
                         "embedding": [float("nan")]}], "usage": {"prompt_tokens": 1}})
    except ValidationError:
        pass
    else:
        raise AssertionError("non-finite embedding accepted")

    matrix = validate_response(
        "token_classify", {"data": [{"index": 0, "data": [[0.1, 0.2], [0.3, 0.4]]}],
                           "usage": {"prompt_tokens": 8}}, 1, "vllm_token_classify")
    assert matrix["format"] == "vllm_token_classify_matrix"
    assert matrix["matrix_shape"] == [2, 2]
    for bad in (
            {"data": [{"data": [0.1, 0.2]}], "usage": {"prompt_tokens": 8}},
            {"data": [{"data": [[float("nan"), 0.2]]}], "usage": {"prompt_tokens": 8}},
            {"data": [{"data": [[0.1, 0.2]]}]}):
        try:
            validate_response("token_classify", bad, 1, "vllm_token_classify")
        except ValidationError:
            pass
        else:
            raise AssertionError(f"invalid token_classify matrix accepted: {bad}")

    dense = [0.0] * 1001
    dense[1], dense[1000] = 0.5, 1.5
    dense_body = {"data": [{"index": 0, "embedding": dense}],
                  "usage": {"prompt_tokens": 8, "total_tokens": 8}}
    dense_validated = validate_response("sparse", dense_body, 1, "vllm_sparse")
    assert dense_validated["format"] == "dense_vocab_splade"
    assert dense_validated["dense_vocab_size"] == 1001
    assert dense_validated["nonzero_count"] == 2
    assert [pair["index"] for pair in dense_validated["values"]] == [1, 1000]
    negative = list(dense)
    negative[2] = -0.1
    for bad, style in (
            ({"data": [{"index": 0, "embedding": [0.1] * 1000}],
              "usage": {"prompt_tokens": 8}}, "vllm_sparse"),
            ({"data": [{"index": 0, "embedding": negative}],
              "usage": {"prompt_tokens": 8}}, "vllm_sparse"),
            ({"data": [{"index": 0, "embedding": [[0.1] * 1001]}],
              "usage": {"prompt_tokens": 8}}, "vllm_sparse"),
            ({"data": [{"index": 0, "embedding": dense}]}, "vllm_sparse"),
            (dense_body, None)):
        try:
            validate_response("sparse", bad, 1, style)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"invalid sparse body accepted for style {style}")

    mock = asyncio.run(_mock_pipeline_test(
        tokenizer, safe, specials, args.tokenizer))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.json"
        _write_json_atomic(path, {"ok": True, "values": [1.0]})
        assert json.loads(path.read_text())["ok"] is True
    print(f"self-tests: ok ({len(safe)} safe words, {specials} specials, "
          f"pool {POOL_SIZE}; {mock})")
    return 0


def _parse_lengths(value):
    try:
        lengths = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers") from exc
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be positive integers")
    return lengths


def _parser():
    parser = argparse.ArgumentParser(
        description="Closed-loop cross-runtime HTTP benchmark client "
                    "(numerical smoke, not task accuracy).")
    parser.add_argument("--url")
    parser.add_argument("--runtime", choices=("infinity", "vllm", "sglang"))
    parser.add_argument("--model", help="served model alias")
    parser.add_argument("--tokenizer", help="local Hugging Face tokenizer directory")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--output", help="absolute output JSON path")
    parser.add_argument("--lengths", type=_parse_lengths,
                        help="task lengths; defaults: 128,512 (image 256,1024; "
                             "decision 8,128,512,2048)")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=float, default=2.0,
                        help="warm-up seconds per workload")
    parser.add_argument("--min-latency-length", type=int, default=8)
    parser.add_argument("--self-test", action="store_true",
                        help="run exact-length, timing, and response-validation tests")
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.self_test:
        if not args.tokenizer:
            parser.error("--self-test requires --tokenizer")
        return _self_test(args)
    for name in ("url", "runtime", "model", "tokenizer", "task", "output"):
        if not getattr(args, name):
            parser.error(f"--{name} is required")
    if not os.path.isabs(args.output):
        parser.error("--output must be an absolute path")
    if args.concurrency < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("--concurrency and --repeats must be >= 1; --warmup must be >= 0")
    if args.seconds <= 0 or args.min_latency_length < 1:
        parser.error("--seconds must be > 0; --min-latency-length must be >= 1")
    if args.lengths is None:
        args.lengths = {"image": [256, 1024], "decision": [8, 128, 512, 2048]}.get(
            args.task, [128, 512])
    try:
        tokenizer = _load_tokenizer(args.tokenizer)
        document = asyncio.run(_benchmark(
            args, tokenizer, _safe_words(tokenizer), _token_len(tokenizer, "")))
    except Exception as exc:
        document = {
            "schema_version": 1, "kind": "cross_runtime_http_numerical_smoke",
            "ok": False, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(args), "workloads": [], "latency_probe": None,
            "runtime": {"requested": args.runtime, "url": args.url, "model": args.model,
                        "task": args.task},
            "client": {"python": platform.python_version(), "platform": platform.platform(),
                       "hostname": socket.gethostname(), "pid": os.getpid(),
                       "http_client": "aiohttp" if aiohttp is not None else "httpx2",
                       "numpy": np.__version__, "transformers": transformers.__version__,
                       "tokenizer": {"path": args.tokenizer}},
            "cache_control": _cache_control(),
            "failures": [{"phase": "fatal", "detail": repr(exc)}]}
    _write_json_atomic(args.output, document)
    return 0 if document["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
