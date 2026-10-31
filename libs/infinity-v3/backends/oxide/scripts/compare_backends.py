#!/usr/bin/env python3
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""Compare two already-running BERT servers; this is not a benchmark.

Start both servers with the same checkpoint and --pooling setting, then run
this script separately for mean, cls and last_token. No models are downloaded
and no server processes are started by this script.
"""

import argparse
import base64
import json
import math
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request

MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class ParityError(Exception):
    pass


def request(url, path, timeout, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ParityError("response exceeds 16 MiB")
    return json.loads(raw)


def vector(value):
    if isinstance(value, dict):
        shape = value.get("shape")
        if (
            value.get("dtype") != "float32"
            or value.get("data_encoding") != "base64"
            or not isinstance(shape, list)
            or len(shape) != 1
            or type(shape[0]) is not int
            or shape[0] <= 0
            or shape[0] > MAX_RESPONSE_BYTES // 4
            or not isinstance(value.get("data"), str)
        ):
            raise ParityError("invalid pooled float32 tensor envelope")
        try:
            raw = base64.b64decode(value["data"], validate=True)
        except ValueError as error:
            raise ParityError("invalid tensor base64") from error
        if len(raw) != shape[0] * 4:
            raise ParityError("tensor shape and byte length disagree")
        # Infinity's tensor wire contract is explicitly little-endian.
        values = [x[0] for x in struct.iter_unpack("<f", raw)]
    elif isinstance(value, list) and value:
        if any(type(x) not in (int, float) for x in value):
            raise ParityError("embedding must contain only numbers")
        values = [float(x) for x in value]
    else:
        raise ParityError("expected a nonempty vector or pooled tensor")
    if not all(math.isfinite(x) for x in values):
        raise ParityError("nonfinite embedding")
    return values


def embeddings(response, count):
    if isinstance(response, dict):
        rows = response.get("data")
        if not isinstance(rows, list) or len(rows) != count:
            raise ParityError("wrong OpenAI response batch size")
        ordered = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ParityError("invalid OpenAI response row")
            index = row.get("index")
            if type(index) is not int or not 0 <= index < count or index in ordered:
                raise ParityError("missing, duplicate or invalid response index")
            ordered[index] = vector(row.get("embedding"))
        return [ordered[i] for i in range(count)]
    if not isinstance(response, list) or len(response) != count:
        raise ParityError("wrong embedding response batch size")
    return [vector(row) for row in response]


def compare(a, b, min_cosine, max_relative_l2):
    if len(a) != len(b):
        raise ParityError(f"dimension mismatch: {len(a)} vs {len(b)}")
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        raise ParityError("zero-norm embedding cannot establish cosine parity")
    cosine = math.fsum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
    relative_l2 = math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b))) / norm_a
    if not math.isfinite(cosine) or not math.isfinite(relative_l2):
        raise ParityError("nonfinite comparison metrics")
    if cosine < min_cosine or relative_l2 > max_relative_l2:
        raise ParityError(f"numerical mismatch: cosine={cosine:.9f}, relative_l2={relative_l2:.9g}")
    return cosine, relative_l2, max(abs(x - y) for x, y in zip(a, b))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--cuda-url", required=True)
    parser.add_argument("--pooling", required=True, type=lambda s: s.replace("-", "_"),
                        choices=["mean", "cls", "last_token"])
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--model", help="common served-model alias, when required")
    parser.add_argument("--encoding-format", choices=["tensor", "float"], default="tensor")
    parser.add_argument("--endpoint", choices=["/embed", "/v1/embeddings"], default="/embed",
                        help="/embed checks unnormalized output; OpenAI always normalizes")
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--min-cosine", type=float,
                        help="default: 0.999 for BF16, 0.99999 for FP32")
    parser.add_argument("--max-relative-l2", type=float,
                        help="default: 0.03 for BF16, 0.0005 for FP32")
    args = parser.parse_args()
    if args.min_cosine is None:
        args.min_cosine = 0.999 if args.dtype == "bfloat16" else 0.99999
    if args.max_relative_l2 is None:
        args.max_relative_l2 = 0.03 if args.dtype == "bfloat16" else 0.0005
    for url in [args.baseline_url, args.cuda_url]:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
            parser.error("URLs must be HTTP(S) server roots without query or fragment")
    if args.baseline_url.rstrip("/") == args.cuda_url.rstrip("/"):
        parser.error("baseline and CUDA URLs must be different")
    if args.vocab_size < 2 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("vocab size must be at least 2 and timeout must be positive")
    if not math.isfinite(args.min_cosine) or not 0 < args.min_cosine <= 1:
        parser.error("min cosine must be in (0, 1]")
    if not math.isfinite(args.max_relative_l2) or args.max_relative_l2 <= 0:
        parser.error("max relative L2 must be positive")
    return args


def main():
    args = arguments()
    info = [request(url, "/info", args.timeout) for url in [args.baseline_url, args.cuda_url]]
    for item in info:
        if not isinstance(item, dict) or item.get("model_dtype") != args.dtype:
            raise ParityError("both servers must use the requested dtype")
        if item.get("enable_fp8_dynamic", False):
            raise ParityError("disable dynamic FP8 on both servers for this comparison")
        model_type = item.get("model_type")
        embedding = model_type.get("embedding") if isinstance(model_type, dict) else None
        if not isinstance(embedding, dict) or embedding.get("pooling") != args.pooling:
            raise ParityError("both servers must use the requested pooling configuration")
        maximum = item.get("max_input_length")
        if type(maximum) is not int or maximum < 512:
            raise ParityError("both servers must admit 512-token inputs")
    a_sha, b_sha = info[0].get("model_sha"), info[1].get("model_sha")
    if a_sha and b_sha and a_sha != b_sha:
        raise ParityError("checkpoint revisions differ")
    if info[0].get("model_id") != info[1].get("model_id") and not (a_sha and a_sha == b_sha):
        raise ParityError("checkpoint IDs differ and no matching revision proves their identity")
    report = {"pooling": args.pooling, "endpoint": args.endpoint,
              "dtype": args.dtype, "min_cosine_threshold": args.min_cosine,
              "max_relative_l2_threshold": args.max_relative_l2,
              "encoding_format": args.encoding_format, "baseline_info": info[0],
              "cuda_info": info[1], "cases": []}
    cases = [(f"{length}x{batch}", [length] * batch) for length in [128, 512] for batch in [1, 8]]
    cases.append(("ragged", [7, 31, 64, 127, 128, 255, 384, 512]))
    dimension = None
    for name, lengths in cases:
        ids = [[1 + (column * 13 + row * 31) % (args.vocab_size - 1) for column in range(length)]
               for row, length in enumerate(lengths)]
        body = {"encoding_format": args.encoding_format}
        body["inputs" if args.endpoint == "/embed" else "input"] = ids
        if args.endpoint == "/embed":
            body.update(normalize=False, truncate=False)
        if args.model:
            body["model"] = args.model
        outputs = [embeddings(request(url, args.endpoint, args.timeout, body), len(ids))
                   for url in [args.baseline_url, args.cuda_url]]
        rows = []
        for index, (a, b) in enumerate(zip(*outputs)):
            try:
                cosine, relative_l2, max_abs = compare(a, b, args.min_cosine, args.max_relative_l2)
            except ParityError as error:
                raise ParityError(f"{name}, row {index}: {error}") from error
            if dimension is not None and len(a) != dimension:
                raise ParityError("embedding dimensions changed between shapes")
            dimension = len(a)
            rows.append((cosine, relative_l2, max_abs))
        report["cases"].append({"name": name, "lengths": lengths, "dimensions": dimension,
                                "min_cosine": min(row[0] for row in rows),
                                "max_relative_l2": max(row[1] for row in rows),
                                "max_abs": max(row[2] for row in rows)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ParityError, OSError, ValueError, urllib.error.HTTPError) as error:
        print(f"Parity check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
