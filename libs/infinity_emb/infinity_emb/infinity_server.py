import concurrent.futures
import time

import typer
import uvicorn
from fastapi import FastAPI, responses, status
from prometheus_fastapi_instrumentator import Instrumentator

# prometheus
import infinity_emb
from infinity_emb.fastapi_schemas import docs, errors
from infinity_emb.fastapi_schemas.convert import list_embeddings_to_response
from infinity_emb.fastapi_schemas.pymodels import (
    OpenAIEmbeddingInput,
    OpenAIEmbeddingResult,
    OpenAIModelInfo,
)
from infinity_emb.inference import BatchHandler, models, select_model_to_functional
from infinity_emb.log_handler import UVICORN_LOG_LEVELS, logger


def create_server(
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    url_prefix: str = "/v1",
    batch_size: int = 64,
    engine: models.InferenceEngine = models.InferenceEngine.torch,
    verbose: bool = False,
    model_warmup=True,
    doc_extra: dict = {},
):
    app = FastAPI(
        title=docs.FASTAPI_TITLE,
        summary=docs.FASTAPI_SUMMARY,
        description=docs.FASTAPI_DESCRIPTION,
        version=infinity_emb.__version__,
        contact=dict(name="Michael Feil"),
        docs_url="/docs",
        license_info={
            "name": "MIT License",
            "identifier": "MIT",
        },
    )
    instrumentator = Instrumentator().instrument(app)
    app.add_exception_handler(errors.OpenAIException, errors.openai_exception_handler)

    @app.on_event("startup")
    async def _startup():
        instrumentator.expose(app)

        model = select_model_to_functional(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            engine=engine,
            model_warmup=model_warmup,
        )

        app.tp = concurrent.futures.ThreadPoolExecutor()
        app.batch_handler = BatchHandler(
            max_batch_size=batch_size, model=model, threadpool=app.tp, verbose=verbose
        )
        # start in a threadpool
        await app.batch_handler.spawn()

        logger.info(
            docs.startup_message(
                host=doc_extra.pop("host", "localhost"),
                port=doc_extra.pop("port", "PORT"),
                prefix=url_prefix,
            )
        )

    @app.on_event("shutdown")
    async def _shutdown():
        app.batch_handler.shutdown()
        app.tp.shutdown()

    @app.get("/ready")
    async def _ready() -> float:
        if app.batch_handler.ready():  # type: ignore
            return time.time()
        else:
            raise errors.OpenAIException(
                "model not ready", code=status.HTTP_503_SERVICE_UNAVAILABLE
            )

    @app.get(
        f"{url_prefix}/models",
        response_model=OpenAIModelInfo,
        response_class=responses.ORJSONResponse,
    )
    async def _models():
        """get models endpoint"""
        s = app.batch_handler.overload_status()  # type: ignore
        return dict(
            data=dict(
                id=model_name_or_path,
                stats=dict(
                    queue_fraction=s.queue_fraction,
                    queue_absolute=s.queue_absolute,
                    results_pending=s.results_absolute,
                    batch_size=batch_size,
                ),
            )
        )

    @app.post(
        f"{url_prefix}/embeddings",
        response_model=OpenAIEmbeddingResult,
        response_class=responses.ORJSONResponse,
    )
    async def _embeddings(data: OpenAIEmbeddingInput):
        """Encode Embeddings

        ```python
        import requests
        requests.post("https://..:8000/v1/embeddings",
            json={"model":"all-MiniLM-L6-v2","input":["A sentence to encode."]})
        """
        bh: BatchHandler = app.batch_handler
        if bh.is_overloaded():
            raise errors.OpenAIException(
                "model overloaded", code=status.HTTP_429_TOO_MANY_REQUESTS
            )

        try:
            logger.debug("[📝] Received request with %s inputs ", len(data.input))
            start = time.perf_counter()

            embedding, usage = await bh.schedule(data.input)

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            res = list_embeddings_to_response(
                embeddings=embedding, model=data.model, usage=usage
            )

            return res
        except Exception as ex:
            raise errors.OpenAIException(
                f"internal server error {ex}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    return app


def start_uvicorn(
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    url_prefix: str = "/v1",
    host: str = "0.0.0.0",
    port: int = 8001,
    log_level: UVICORN_LOG_LEVELS = UVICORN_LOG_LEVELS.info.name,  # type: ignore
    engine: models.InferenceEngineTypeHint = models.InferenceEngineTypeHint.torch.name,  # type: ignore # noqa
    model_warmup: bool = True,
):
    """Infinity Embedding API ♾️  cli to start a uvicorn-server instance;
    MIT License; Copyright (c) 2023 Michael Feil

    Args:
        model_name_or_path: str: Huggingface model, e.g.
            "sentence-transformers/all-MiniLM-L6-v2".
        batch_size: int: batch size for forward pass.
        url_prefix str: prefix for api. typically "/v1".
        host str: host-url, typically either "0.0.0.0" or "127.0.0.1".
        port int: port that you want to expose.
        log_level: logging level.
            For high performance, use "info" or higher levels. Defaults to "info".
        engine: framework that should perform inference.
        model_warmup: perform model warmup before starting the server. Defaults to True.
    """
    engine_load: models.InferenceEngine = models.InferenceEngine[engine.name]
    logger.setLevel(log_level.to_int())

    app = create_server(
        model_name_or_path=model_name_or_path,
        url_prefix=url_prefix,
        batch_size=batch_size,
        engine=engine_load,
        verbose=log_level.to_int() <= 10,
        doc_extra=dict(host=host, port=port),
        model_warmup=model_warmup,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level.name)


def cli():
    """fires the command line using Python `typer.run()`"""
    typer.run(start_uvicorn)


# app = create_server()
if __name__ == "__main__":
    # for debugging
    cli()
