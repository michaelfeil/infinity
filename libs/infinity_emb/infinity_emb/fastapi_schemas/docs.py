FASTAPI_TITLE = "♾️ Infinity - Embedding Inference Server"
FASTAPI_SUMMARY = "Embedding Inference Server - finding TGI for embeddings"


def startup_message(host: str, port: str, prefix: str) -> str:
    return f"""

♾️  Infinity - Embedding Inference Server
MIT License; Copyright (c) 2023 Michael Feil

Open the Docs via Swagger UI:
http://{host}:{port}/docs

Access model via 'GET':
curl http://{host}:{port}{prefix}/models
"""
