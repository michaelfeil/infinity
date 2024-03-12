import time
from typing import Optional

# prometheus
import infinity_emb
from infinity_emb.args import EngineArgs
from infinity_emb.engine import AsyncEmbeddingEngine
from infinity_emb.fastapi_schemas import docs, errors
from infinity_emb.fastapi_schemas.convert import (
    list_embeddings_to_response,
    to_rerank_response,
)
from infinity_emb.fastapi_schemas.pymodels import (
    OpenAIEmbeddingInput,
    OpenAIEmbeddingResult,
    OpenAIModelInfo,
    RerankInput,
)
from infinity_emb.inference import Device, DeviceTypeHint
from infinity_emb.inference.caching_layer import INFINITY_CACHE_VECTORS
from infinity_emb.log_handler import UVICORN_LOG_LEVELS, logger
from infinity_emb.primitives import InferenceEngine, InferenceEngineTypeHint


def create_server(
    engine_args: EngineArgs,
    url_prefix: str = "",
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

    model_name_response_name = "".join(engine_args.model_name_or_path.split("/")[-2:])

    @app.on_event("startup")
    async def _startup():
        instrumentator.expose(app)

        app.model = AsyncEmbeddingEngine.from_args(engine_args)
        # start in a threadpool
        await app.model.astart()

        logger.info(
            docs.startup_message(
                host=doc_extra.pop("host", "localhost"),
                port=doc_extra.pop("port", "PORT"),
                prefix=url_prefix,
            )
        )

    @app.on_event("shutdown")
    async def _shutdown():
        await app.model.astop()

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
        s = app.model.overload_status()  # type: ignore
        return dict(
            data=dict(
                id=engine_args.model_name_or_path,
                stats=dict(
                    queue_fraction=s.queue_fraction,
                    queue_absolute=s.queue_absolute,
                    results_pending=s.results_absolute,
                    batch_size=engine_args.batch_size,
                ),
                backend=engine_args.engine.name,
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
        requests.post("http://..:7997/embeddings",
            json={"model":"bge-small-en-v1.5","input":["A sentence to encode."]})
        """
        model: AsyncEmbeddingEngine = app.model  # type: ignore
        if model.is_overloaded():
            raise errors.OpenAIException(
                "model overloaded", code=status.HTTP_429_TOO_MANY_REQUESTS
            )

        try:
            if isinstance(data.input, str):
                data.input = [data.input]

            logger.debug("[📝] Received request with %s inputs ", len(data.input))
            start = time.perf_counter()

            embedding, usage = await model.embed(data.input)

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            res = list_embeddings_to_response(
                embeddings=embedding, model=model_name_response_name, usage=usage
            )

            return res
        except Exception as ex:
            raise errors.OpenAIException(
                f"InternalServerError: {ex}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @app.post(
        f"{url_prefix}/rerank",
        # response_model=RerankResult,
        response_class=responses.ORJSONResponse,
    )
    async def _rerank(data: RerankInput):
        """Encode Embeddings

        ```python
        import requests
        requests.post("http://..:7997/rerank",
            json={"query":"Where is Munich?","texts":["Munich is in Germany."]})
        """
        model: AsyncEmbeddingEngine = app.model  # type: ignore
        if model.is_overloaded():
            raise errors.OpenAIException(
                "model overloaded", code=status.HTTP_429_TOO_MANY_REQUESTS
            )

        try:
            logger.debug("[📝] Received request with %s docs ", len(data.documents))
            start = time.perf_counter()

            scores, usage = await model.rerank(
                query=data.query, docs=data.documents, raw_scores=False
            )

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            if data.return_documents:
                docs = data.documents
            else:
                docs = None

            res = to_rerank_response(
                scores=scores,
                documents=docs,
                model=model_name_response_name,
                usage=usage,
            )

            return res
        except Exception as ex:
            raise errors.OpenAIException(
                f"InternalServerError: {ex}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    return app


def _start_uvicorn(
    model_name_or_path: str = "michaelfeil/bge-small-en-v1.5",
    batch_size: int = 64,
    revision: Optional[str] = None,
    trust_remote_code: bool = True,
    url_prefix: str = "",
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
    MIT License; Copyright (c) 2023-now Michael Feil

    Args:
        model_name_or_path, str: Huggingface model, e.g.
            "michaelfeil/bge-small-en-v1.5".
        batch_size, int: batch size for forward pass.
        revision: str: revision of the model.
        trust_remote_code, bool: trust remote code.
        url_prefix, str: prefix for api. typically "".
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

    logger.setLevel(log_level.to_int())

    vector_disk_cache_path = (
        f"{engine}_{model_name_or_path.replace('/','_')}" if vector_disk_cache else ""
    )

    engine_args = EngineArgs(
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
        revision=revision,
        trust_remote_code=trust_remote_code,
        engine=InferenceEngine[engine.value],  # type: ignore
        model_warmup=model_warmup,
        vector_disk_cache_path=vector_disk_cache_path,
        device=Device[device.value],  # type: ignore
        lengths_via_tokenize=lengths_via_tokenize,
    )

    app = create_server(
        engine_args,
        url_prefix=url_prefix,
        doc_extra=dict(host=host, port=port),
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level.name)


def cli():
    """fires the command line using Python `typer.run()`"""
    import typer

    typer.run(_start_uvicorn)


if __name__ == "__main__":
    # for debugging
    cli()
