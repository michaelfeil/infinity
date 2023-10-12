# from infinity_emb import __version__

FASTAPI_TITLE = "♾️ Infinity - Embedding Inference Server"
FASTAPI_SUMMARY = "Embedding Inference Server - finding TGI for embeddings"
FASTAPI_DESCRIPTION = ""


def startup_message(host: str, port: str, prefix: str) -> str:
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
