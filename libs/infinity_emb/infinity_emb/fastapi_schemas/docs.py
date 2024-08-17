# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

FASTAPI_TITLE = "♾️ Infinity - Embedding Inference Server"
FASTAPI_SUMMARY = (
    "Infinity is a high-throughput, low-latency REST API for serving text-embeddings, reranking models and clip. "
    "Infinity is developed under MIT License at https://github.com/michaelfeil/infinity."
)
FASTAPI_DESCRIPTION = ""


def startup_message(host: str, port: int, prefix: str) -> str:
    from infinity_emb import __version__

    return f"""

♾️  Infinity - Embedding Inference Server
MIT License; Copyright (c) 2023 Michael Feil
Version {__version__}

Open the Docs via Swagger UI:
http://{host}:{port}/docs

Access model via 'GET':
curl http://{host}:{port}{prefix}/models
"""
