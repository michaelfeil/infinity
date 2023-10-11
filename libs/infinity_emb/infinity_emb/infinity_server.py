import concurrent.futures
import time

import typer
import uvicorn
from fastapi import FastAPI, status
from prometheus_fastapi_instrumentator import Instrumentator

# prometheus
import infinity_emb
from infinity_emb.fastapi_schemas import errors
from infinity_emb.fastapi_schemas.convert import list_embeddings_to_response
from infinity_emb.fastapi_schemas.pymodels import (
    ModelInfo,
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
):
    app = FastAPI(
        title="♾️ Infinity - Embedding Inference Server",
        summary="Embedding Inference Server - finding TGI for embeddings",
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
            model_name_or_path=model_name_or_path, batch_size=batch_size, engine=engine
        )

        app.tp = concurrent.futures.ThreadPoolExecutor()
        app.batch_handler = BatchHandler(
            max_batch_size=batch_size, model=model, threadpool=app.tp, verbose=verbose
        )
        app.tokenize_len = model.tokenize_lengths
        # start in a threadpool
        await app.batch_handler.spawn()

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

    @app.get(f"{url_prefix}/models")
    async def _models() -> OpenAIModelInfo:
        """get models endpoint"""
        s = app.batch_handler.overload_status()  # type: ignore
        return OpenAIModelInfo(
            data=ModelInfo(
                id=model_name_or_path,
                stats=dict(
                    queue_fraction=s.queue_fraction,
                    queue_absolute=s.queue_absolute,
                    results_pending=s.results_absolute,
                ),
            )
        )

    @app.post(f"{url_prefix}/embeddings")
    async def _embeddings(data: OpenAIEmbeddingInput) -> OpenAIEmbeddingResult:
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
            start = time.perf_counter()

            # lengths, usage = await to_thread(
            #   models.get_lengths_with_tokenize, app.tp, data.input, app.tokenize_len)
            lengths, usage = models.get_lengths_with_tokenize(
                data.input  # , app.tokenize_len
            )
            logger.debug("[📝] Received request with %s inputs ", len(lengths))

            # emb = await asyncio.gather(
            #     *[(bh.schedule(s, prio=prio)) for s, prio in zip(data.input, lengths)]
            # )
            emb = await bh.schedule(data.input, prios=lengths)

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            res = list_embeddings_to_response(
                embeddings=emb, model=data.model, usage=usage
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
    """
    engine_load: models.InferenceEngine = models.InferenceEngine[engine.name]
    logger.setLevel(log_level.to_int())

    app = create_server(
        model_name_or_path=model_name_or_path,
        url_prefix=url_prefix,
        batch_size=batch_size,
        engine=engine_load,
        verbose=log_level.to_int() <= 10,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level.name)


def cli():
    """fires the command line using Python `typer.run()`"""
    typer.run(start_uvicorn)


if __name__ == "__main__":
    # for debugging
    cli()
