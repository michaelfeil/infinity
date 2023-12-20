import time

# prometheus
import infinity_emb
from infinity_emb.fastapi_schemas import docs, errors
from infinity_emb.fastapi_schemas.convert import list_embeddings_to_response
from infinity_emb.fastapi_schemas.pymodels import (
    OpenAIEmbeddingInput,
    OpenAIEmbeddingResult,
    OpenAIModelInfo,
)
from infinity_emb.inference import (
    BatchHandler,
    Device,
    DeviceTypeHint,
    select_model,
)
from infinity_emb.inference.caching_layer import INFINITY_CACHE_VECTORS
from infinity_emb.log_handler import UVICORN_LOG_LEVELS, logger
from infinity_emb.transformer.utils import InferenceEngine, InferenceEngineTypeHint


def create_server(
    model_name_or_path: str = "BAAI/bge-small-en-v1.5",
    url_prefix: str = "/v1",
    batch_size: int = 64,
    engine: InferenceEngine = InferenceEngine.torch,
    verbose: bool = False,
    model_warmup=True,
    vector_disk_cache=INFINITY_CACHE_VECTORS,
    device: Device = Device.auto,
    lengths_via_tokenize: bool = False,
    doc_extra: dict = {},
):
    """
    creates the FastAPI App
    """
    from fastapi import FastAPI, responses, status
    from prometheus_fastapi_instrumentator import Instrumentator

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
    vector_disk_cache_path = (
        f"{engine}_{model_name_or_path.replace('/','_')}" if vector_disk_cache else ""
    )

    @app.on_event("startup")
    async def _startup():
        instrumentator.expose(app)

        model, min_inference_t = select_model(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            engine=engine,
            model_warmup=model_warmup,
            device=device,
        )

        app.batch_handler = BatchHandler(
            max_batch_size=batch_size,
            model=model,
            verbose=verbose,
            batch_delay=min_inference_t / 2,
            vector_disk_cache_path=vector_disk_cache_path,
            lengths_via_tokenize=lengths_via_tokenize,
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
        await app.batch_handler.shutdown()

    @app.get("/ready")
    async def _ready() -> float:
        """
        returns always the current time
        """
        return time.time()

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
                backend=engine.name,
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
        requests.post("http://..:7997/v1/embeddings",
            json={"model":"bge-small-en-v1.5","input":["A sentence to encode."]})
        """
        bh: BatchHandler = app.batch_handler  # type: ignore
        if bh.is_overloaded():
            raise errors.OpenAIException(
                "model overloaded", code=status.HTTP_429_TOO_MANY_REQUESTS
            )

        try:
            logger.debug("[📝] Received request with %s inputs ", len(data.input))
            start = time.perf_counter()

            embedding, usage = await bh.embed(data.input)

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


def _start_uvicorn(
    model_name_or_path: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    url_prefix: str = "/v1",
    host: str = "0.0.0.0",
    port: int = 7997,
    log_level: UVICORN_LOG_LEVELS = UVICORN_LOG_LEVELS.info.name,  # type: ignore
    engine: InferenceEngineTypeHint = InferenceEngineTypeHint.torch.name,  # type: ignore # noqa
    model_warmup: bool = True,
    vector_disk_cache: bool = INFINITY_CACHE_VECTORS,
    device: DeviceTypeHint = DeviceTypeHint.auto.name,  # type: ignore
    lengths_via_tokenize: bool = False,
):
    """Infinity Embedding API ♾️  cli to start a uvicorn-server instance;
    MIT License; Copyright (c) 2023 Michael Feil

    Args:
        model_name_or_path, str: Huggingface model, e.g.
            "BAAI/bge-small-en-v1.5".
        batch_size, int: batch size for forward pass.
        url_prefix, str: prefix for api. typically "/v1".
        host, str: host-url, typically either "0.0.0.0" or "127.0.0.1".
        port, int: port that you want to expose.
        log_level: logging level.
            For high performance, use "info" or higher levels. Defaults to "info".
        engine, str: framework that should perform inference.
        model_warmup, bool: perform model warmup before starting the server.
            Defaults to True.
        vector_disk_cache, bool: cache past embeddings in SQL.
            Defaults to False or env-INFINITY_CACHE_VECTORS if set
        device, Device: device to use for inference. Defaults to Device.auto or "auto"
        lengths_via_tokenize: bool,
    """
    import uvicorn

    engine_load: InferenceEngine = InferenceEngine[engine.name]
    used_device: Device = Device[device.name]
    logger.setLevel(log_level.to_int())

    app = create_server(
        model_name_or_path=model_name_or_path,
        url_prefix=url_prefix,
        batch_size=batch_size,
        engine=engine_load,
        verbose=log_level.to_int() <= 10,
        doc_extra=dict(host=host, port=port),
        model_warmup=model_warmup,
        vector_disk_cache=vector_disk_cache,
        device=used_device,
        lengths_via_tokenize=lengths_via_tokenize,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level.name)


def cli():
    """fires the command line using Python `typer.run()`"""
    import typer

    typer.run(_start_uvicorn)


# app = create_server()
if __name__ == "__main__":
    # for debugging
    cli()
