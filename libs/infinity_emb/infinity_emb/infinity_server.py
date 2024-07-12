import asyncio
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import infinity_emb
from infinity_emb._optional_imports import CHECK_TYPER, CHECK_UVICORN
from infinity_emb.args import EngineArgs
from infinity_emb.engine import AsyncEmbeddingEngine, AsyncEngineArray
from infinity_emb.env import MANAGER
from infinity_emb.fastapi_schemas import docs, errors
from infinity_emb.fastapi_schemas.pymodels import (
    ClassifyInput,
    ClassifyResult,
    ImageEmbeddingInput,
    OpenAIEmbeddingInput,
    OpenAIEmbeddingResult,
    OpenAIModelInfo,
    RerankInput,
    ReRankResult,
)
from infinity_emb.log_handler import UVICORN_LOG_LEVELS, logger
from infinity_emb.primitives import (
    Device,
    Dtype,
    EmbeddingDtype,
    ImageCorruption,
    InferenceEngine,
    ModelNotDeployedError,
    PoolingMethod,
)


def create_server(
    *,
    engine_args_list: list[EngineArgs],
    url_prefix: str = MANAGER.url_prefix,
    doc_extra: dict[str, Any] = {},
    redirect_slash: str = MANAGER.redirect_slash,
    preload_only: bool = MANAGER.preload_only,
    permissive_cors: bool = MANAGER.permissive_cors,
    api_key: str = MANAGER.api_key,
    proxy_root_path: str = MANAGER.proxy_root_path,
):
    """
    creates the FastAPI App
    """
    from fastapi import Depends, FastAPI, HTTPException, responses, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from prometheus_fastapi_instrumentator import Instrumentator

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        instrumentator.expose(app)  # type: ignore
        app.engine_array = AsyncEngineArray.from_args(engine_args_list)  # type: ignore
        # start in a threadpool
        await app.engine_array.astart()  # type: ignore

        logger.info(
            docs.startup_message(
                host=doc_extra.pop("host", None),
                port=doc_extra.pop("port", None),
                prefix=url_prefix,
            )
        )

        if preload_only:
            # this is a hack to exit the process after 3 seconds
            async def kill_later(seconds: int):
                await asyncio.sleep(seconds)
                os.kill(os.getpid(), signal.SIGINT)

            logger.info(
                f"Preloaded configuration successfully. {engine_args_list} "
                " -> exit ."
            )
            asyncio.create_task(kill_later(3))
        yield
        await app.engine_array.astop()  # type: ignore
        # shutdown!

    app = FastAPI(
        title=docs.FASTAPI_TITLE,
        summary=docs.FASTAPI_SUMMARY,
        description=docs.FASTAPI_DESCRIPTION,
        version=infinity_emb.__version__,
        contact=dict(name="Michael Feil"),
        docs_url=f"{url_prefix}/docs",
        openapi_url=f"{url_prefix}/openapi.json",
        license_info={
            "name": "MIT License",
            "identifier": "MIT",
        },
        lifespan=lifespan,
        root_path=proxy_root_path,
    )
    route_dependencies = []

    if permissive_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    if api_key:
        oauth2_scheme = HTTPBearer(auto_error=False)

        async def validate_token(
            credential: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
        ):
            if credential is None or credential.credentials != api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        route_dependencies.append(Depends(validate_token))

    instrumentator = Instrumentator().instrument(app)
    app.add_exception_handler(errors.OpenAIException, errors.openai_exception_handler)

    @app.get("/health", operation_id="health", response_class=responses.ORJSONResponse)
    async def _health() -> dict[str, float]:
        """
        health check endpoint

        Returns:
            dict(unix=float): dict with unix time stamp
        """
        return {"unix": time.time()}

    if redirect_slash:
        from fastapi.responses import RedirectResponse

        assert redirect_slash.startswith("/"), "redirect_slash must start with /"

        @app.get("/")
        async def redirect():
            response = RedirectResponse(url=redirect_slash)
            return response

    @app.get(
        f"{url_prefix}/models",
        response_model=OpenAIModelInfo,
        response_class=responses.ORJSONResponse,
        dependencies=route_dependencies,
        operation_id="models",
    )
    async def _models():
        """get models endpoint"""
        engine_array: "AsyncEngineArray" = app.engine_array  # type: ignore
        data = []
        for engine in engine_array:
            engine_args = engine.engine_args
            data.append(
                dict(
                    id=engine_args.served_model_name,
                    stats=dict(
                        queue_fraction=engine.overload_status().queue_fraction,
                        queue_absolute=engine.overload_status().queue_absolute,
                        results_pending=engine.overload_status().results_absolute,
                        batch_size=engine_args.batch_size,
                    ),
                    capabilities=engine.capabilities,
                    backend=engine_args.engine.name,
                    embedding_dtype=engine_args.embedding_dtype.name,
                    dtype=engine_args.dtype.name,
                    revision=engine_args.revision,
                    lengths_via_tokenize=engine_args.lengths_via_tokenize,
                    device=engine_args.device.name,
                )
            )

        return dict(data=data)

    def _resolve_engine(model: str) -> "AsyncEmbeddingEngine":
        try:
            engine: "AsyncEmbeddingEngine" = app.engine_array[model]  # type: ignore
        except IndexError as ex:
            raise errors.OpenAIException(
                f"Invalid model: {ex}",
                code=status.HTTP_400_BAD_REQUEST,
            )
        if engine.is_overloaded():
            raise errors.OpenAIException(
                f"model {model} is currently overloaded",
                code=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        return engine

    @app.post(
        f"{url_prefix}/embeddings",
        response_model=OpenAIEmbeddingResult,
        response_class=responses.ORJSONResponse,
        dependencies=route_dependencies,
        operation_id="embeddings",
    )
    async def _embeddings(data: OpenAIEmbeddingInput):
        """Encode Embeddings

        ```python
        import requests
        requests.post("http://..:7997/embeddings",
            json={"model":"BAAI/bge-small-en-v1.5","input":["A sentence to encode."]})
        """
        engine = _resolve_engine(data.model)

        try:
            if isinstance(data.input, str):
                data.input = [data.input]

            logger.debug("[📝] Received request with %s inputs ", len(data.input))
            start = time.perf_counter()

            embedding, usage = await engine.embed(sentences=data.input)

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            return OpenAIEmbeddingResult.to_embeddings_response(
                embeddings=embedding,
                model=engine.engine_args.served_model_name,
                usage=usage,
            )
        except ModelNotDeployedError as ex:
            raise errors.OpenAIException(
                f"ModelNotDeployedError: model=`{data.model}` does not support `embed`. Reason: {ex}",
                code=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as ex:
            raise errors.OpenAIException(
                f"InternalServerError: {ex}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @app.post(
        f"{url_prefix}/rerank",
        response_model=ReRankResult,
        response_class=responses.ORJSONResponse,
        dependencies=route_dependencies,
        operation_id="rerank",
    )
    async def _rerank(data: RerankInput):
        """Rerank documents

        ```python
        import requests
        requests.post("http://..:7997/rerank",
            json={
                "model":"mixedbread-ai/mxbai-rerank-xsmall-v1",
                "query":"Where is Munich?",
                "documents":["Munich is in Germany.", "The sky is blue."]
            })
        """
        engine = _resolve_engine(data.model)
        try:
            logger.debug("[📝] Received request with %s docs ", len(data.documents))
            start = time.perf_counter()

            scores, usage = await engine.rerank(
                query=data.query, docs=data.documents, raw_scores=False
            )

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            if data.return_documents:
                docs = data.documents
            else:
                docs = None

            return ReRankResult.to_rerank_response(
                scores=scores,
                documents=docs,
                model=engine.engine_args.served_model_name,
                usage=usage,
            )
        except ModelNotDeployedError as ex:
            raise errors.OpenAIException(
                f"ModelNotDeployedError: model=`{data.model}` does not support `rerank`. Reason: {ex}",
                code=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as ex:
            raise errors.OpenAIException(
                f"InternalServerError: {ex}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @app.post(
        f"{url_prefix}/classify",
        response_class=responses.ORJSONResponse,
        dependencies=route_dependencies,
        response_model=ClassifyResult,
        operation_id="classify",
    )
    async def _classify(data: ClassifyInput):
        """Score or Classify Sentiments

        ```python
        import requests
        requests.post("http://..:7997/classify",
            json={"model":"SamLowe/roberta-base-go_emotions","input":["I am not having a great day."]})
        """
        engine = _resolve_engine(data.model)
        try:
            logger.debug("[📝] Received request with %s docs ", len(data.input))
            start = time.perf_counter()

            scores_labels, usage = await engine.classify(
                sentences=data.input, raw_scores=data.raw_scores
            )

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            return ClassifyResult.to_classify_response(
                scores_labels=scores_labels,
                model=engine.engine_args.served_model_name,
                usage=usage,
            )
        except ModelNotDeployedError as ex:
            raise errors.OpenAIException(
                f"ModelNotDeployedError: model=`{data.model}` does not support `classify`. Reason: {ex}",
                code=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as ex:
            raise errors.OpenAIException(
                f"InternalServerError: {ex}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @app.post(
        f"{url_prefix}/embeddings_image",
        response_model=OpenAIEmbeddingResult,
        response_class=responses.ORJSONResponse,
        dependencies=route_dependencies,
        operation_id="embeddings_image",
    )
    async def _embeddings_image(data: ImageEmbeddingInput):
        """Encode Embeddings

        ```python
        import requests
        requests.post("http://..:7997/embeddings_image",
            json={"model":"openai/clip-vit-base-patch32","input":["http://images.cocodataset.org/val2017/000000039769.jpg"]})
        """
        engine = _resolve_engine(data.model)
        if hasattr(data.input, "host"):
            # if it is a single url
            urls = [str(data.input)]
        else:
            urls = [str(d) for d in data.input]  # type: ignore
        try:
            logger.debug("[📝] Received request with %s Urls ", len(urls))
            start = time.perf_counter()

            embedding, usage = await engine.image_embed(images=urls)

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            return OpenAIEmbeddingResult.to_embeddings_response(
                embeddings=embedding,
                model=engine.engine_args.served_model_name,
                usage=usage,
            )
        except ImageCorruption as ex:
            raise errors.OpenAIException(
                f"ImageCorruption, could not open {urls} -> {ex}",
                code=status.HTTP_400_BAD_REQUEST,
            )
        except ModelNotDeployedError as ex:
            raise errors.OpenAIException(
                f"ModelNotDeployedError: model=`{data.model}` does not support `embed`. Reason: {ex}",
                code=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as ex:
            raise errors.OpenAIException(
                f"InternalServerError: {ex}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    return app


class AutoPadding:
    """itertools.cycle with custom behaviour"""

    def __init__(self, length: int, **kwargs):
        self.length = length
        self.kwargs = kwargs

    def __call__(self, x):
        """pad x to length of self.length"""
        if not isinstance(x, (list, tuple)):
            return [x] * self.length
        elif len(x) == 1:
            return x * self.length
        elif len(x) == self.length:
            return x
        else:
            raise ValueError(f"Expected length {self.length} but got {len(x)}")

    def __iter__(self):
        """iterate over kwargs and pad them to length of self.length"""
        for iteration in range(self.length):
            kwargs = {}
            for key, value in self.kwargs.items():
                kwargs[key] = self.__call__(value)[iteration]
            yield kwargs


if CHECK_TYPER.is_available:
    CHECK_TYPER.mark_required()
    CHECK_UVICORN.mark_required()
    import typer
    import uvicorn

    tp = typer.Typer()

    @tp.command("v1")
    def v1(
        model_name_or_path: str = MANAGER.model_id[0],
        served_model_name: str = MANAGER.served_model_name[0],
        batch_size: int = MANAGER.batch_size[0],
        revision: str = MANAGER.revision[0],
        trust_remote_code: bool = MANAGER.trust_remote_code[0],
        redirect_slash: str = MANAGER.redirect_slash,
        engine: InferenceEngine = MANAGER.engine[0],  # type: ignore # noqa
        model_warmup: bool = MANAGER.model_warmup[0],
        vector_disk_cache: bool = MANAGER.vector_disk_cache[0],
        device: Device = MANAGER.device[0],  # type: ignore
        lengths_via_tokenize: bool = MANAGER.lengths_via_tokenize[0],
        dtype: Dtype = MANAGER.dtype[0],  # type: ignore
        pooling_method: PoolingMethod = MANAGER.pooling_method[0],  # type: ignore
        compile: bool = MANAGER.compile[0],
        bettertransformer: bool = MANAGER.bettertransformer[0],
        preload_only: bool = MANAGER.preload_only,
        permissive_cors: bool = MANAGER.permissive_cors,
        api_key: str = MANAGER.api_key,
        url_prefix: str = MANAGER.url_prefix,
        host: str = MANAGER.host,
        port: int = MANAGER.port,
        log_level: UVICORN_LOG_LEVELS = MANAGER.log_level,  # type: ignore
    ):
        """Infinity API ♾️ cli v1
        MIT License; Copyright (c) 2023-now Michael Feil

        Args:
            model_name_or_path, str: Huggingface model, e.g.
                "michaelfeil/bge-small-en-v1.5".
            served_model_name, str: "", e.g. "bge-small-en-v1.5"
            batch_size, int: batch size for forward pass.
            revision: str: revision of the model.
            trust_remote_code, bool: trust remote code.
            redirect_slash, str: redirect to of GET "/". Defaults to "/docs". Empty string to disable.
            log_level: logging level.
                For high performance, use "info" or higher levels. Defaults to "info".
            engine, str: framework that should perform inference.
            model_warmup, bool: perform model warmup before starting the server.
                Defaults to True.
            vector_disk_cache, bool: cache past embeddings in SQL.
                Defaults to False
            device, Device: device to use for inference. Defaults to Device.auto or "auto"
            lengths_via_tokenize: bool: schedule by token usage. Defaults to False.
            dtype, Dtype: data type to use for inference. Defaults to Dtype.auto or "auto"
            pooling_method, PoolingMethod: pooling method to use. Defaults to PoolingMethod.auto or "auto"
            compile, bool: compile model for faster inference. Defaults to False.
            use_bettertransformer, bool: use bettertransformer. Defaults to True.
            preload_only, bool: only preload the model and exit. Defaults to False.
            permissive_cors, bool: add permissive CORS headers to enable consumption from a browser. Defaults to False.
            api_key, str: optional Bearer token for authentication. Defaults to "", which disables authentication.
            url_prefix, str: prefix for api. typically "".
            host, str: host-url, typically either "0.0.0.0" or "127.0.0.1".
            port, int: port that you want to expose.
        """
        if api_key:
            raise ValueError(
                "api_key is not supported in `v1`. Please migrate to `v2`."
            )
        logger.warning(
            "CLI v1 is deprecated and might be removed in the future. Please use CLI v2, by specifying `v2` as the command."
        )
        time.sleep(5)
        v2(
            model_id=[model_name_or_path],
            served_model_name=[served_model_name],  # type: ignore
            batch_size=[batch_size],
            revision=[revision],  # type: ignore
            trust_remote_code=[trust_remote_code],
            engine=[engine],
            dtype=[dtype],
            pooling_method=[pooling_method],
            device=[device],
            model_warmup=[model_warmup],
            vector_disk_cache=[vector_disk_cache],
            lengths_via_tokenize=[lengths_via_tokenize],
            compile=[compile],
            bettertransformer=[bettertransformer],
            # unique kwargs
            preload_only=preload_only,
            url_prefix=url_prefix,
            host=host,
            port=port,
            redirect_slash=redirect_slash,
            log_level=log_level,
            permissive_cors=permissive_cors,
            api_key=api_key,
        )

    @tp.command("v2")
    def v2(
        # arguments for engine
        model_id: list[str] = MANAGER.model_id,
        served_model_name: list[str] = MANAGER.served_model_name,
        batch_size: list[int] = MANAGER.batch_size,
        revision: list[str] = MANAGER.revision,
        trust_remote_code: list[bool] = MANAGER.trust_remote_code,
        engine: list[InferenceEngine] = MANAGER.engine,  # type: ignore # noqa
        model_warmup: list[bool] = MANAGER.model_warmup,
        vector_disk_cache: list[bool] = MANAGER.vector_disk_cache,
        device: list[Device] = MANAGER.device,  # type: ignore
        lengths_via_tokenize: list[bool] = MANAGER.lengths_via_tokenize,
        dtype: list[Dtype] = MANAGER.dtype,  # type: ignore
        embedding_dtype: list[EmbeddingDtype] = MANAGER.embedding_dtype,  # type: ignore
        pooling_method: list[PoolingMethod] = MANAGER.pooling_method,  # type: ignore
        compile: list[bool] = MANAGER.compile,
        bettertransformer: list[bool] = MANAGER.bettertransformer,
        # arguments for uvicorn / server
        preload_only: bool = MANAGER.preload_only,
        host: str = MANAGER.host,
        port: int = MANAGER.port,
        url_prefix: str = MANAGER.url_prefix,
        redirect_slash: str = MANAGER.redirect_slash,
        log_level: UVICORN_LOG_LEVELS = MANAGER.log_level,  # type: ignore
        permissive_cors: bool = False,
        api_key: str = MANAGER.api_key,
        proxy_root_path: str = MANAGER.proxy_root_path,
    ):
        """Infinity API ♾️ cli v2
        MIT License; Copyright (c) 2023-now Michael Feil

        kwargs:
            model_id, list[str]: Huggingface model, e.g.
                ["michaelfeil/bge-small-en-v1.5", "mixedbread-ai/mxbai-embed-large-v1"]
            served_model_name, list[str]: "", e.g. ["bge-small-en-v1.5"]
            batch_size, list[int]: batch size for forward pass.
            revision: list[str]: revision of the model.
            trust_remote_code, list[bool]: trust remote code.
            url_prefix, str: prefix for api. typically "".
            host, str: host-url, typically either "0.0.0.0" or "127.0.0.1".
            port, int: port that you want to expose.
            redirect_slash, str: redirect to of GET "/". Defaults to "/docs". Empty string to disable.
            log_level: logging level.
                For high performance, use "info" or higher levels. Defaults to "info".
            engine, str: framework that should perform inference.
            model_warmup, bool: perform model warmup before starting the server.
                Defaults to True.
            vector_disk_cache, bool: cache past embeddings in SQL.
                Defaults to False or env-INFINITY_CACHE_VECTORS if set
            device, Device: device to use for inference. Defaults to Device.auto or "auto"
            lengths_via_tokenize: bool: schedule by token usage. Defaults to False.
            dtype, Dtype: data type to use for inference. Defaults to Dtype.auto or "auto"
            embedding_dtype, EmbeddingDtype: data type to use for embeddings. Defaults to EmbeddingDtype.float32 or "float32"
            pooling_method, PoolingMethod: pooling method to use. Defaults to PoolingMethod.auto or "auto"
            compile, bool: compile model for faster inference. Defaults to False.
            use_bettertransformer, bool: use bettertransformer. Defaults to True.
            preload_only, bool: only preload the model and exit. Defaults to False.
            permissive_cors, bool: add permissive CORS headers to enable consumption from a browser. Defaults to False.
            api_key, str: optional Bearer token for authentication. Defaults to "", which disables authentication.
            proxy_root_path, str: optional Proxy prefix for the application. See: https://fastapi.tiangolo.com/advanced/behind-a-proxy/
        """
        logger.setLevel(log_level.to_int())
        padder = AutoPadding(
            length=len(model_id),
            model_name_or_path=model_id,
            batch_size=batch_size,
            revision=revision,
            trust_remote_code=trust_remote_code,
            engine=engine,
            model_warmup=model_warmup,
            vector_disk_cache_path=vector_disk_cache,
            device=device,
            lengths_via_tokenize=lengths_via_tokenize,
            dtype=dtype,
            embedding_dtype=embedding_dtype,
            pooling_method=pooling_method,
            compile=compile,
            bettertransformer=bettertransformer,
            served_model_name=served_model_name,
        )

        engine_args = []
        for kwargs in padder:
            engine_args.append(EngineArgs(**kwargs))

        app = create_server(
            engine_args_list=engine_args,
            url_prefix=url_prefix,
            doc_extra=dict(host=host, port=port),
            redirect_slash=redirect_slash,
            preload_only=preload_only,
            permissive_cors=permissive_cors,
            api_key=api_key,
            proxy_root_path=proxy_root_path,
        )
        uvicorn.run(app, host=host, port=port, log_level=log_level.name)

    def cli():
        if len(sys.argv) == 1 or sys.argv[1] not in ["v1", "v2", "help", "--help"]:
            for _ in range(3):
                logger.error(
                    "WARNING: No command given. Defaulting to `v1`. "
                    "This will be deprecated in the future, and will require usage of a `v1` or `v2`. "
                    "Specify the version of the CLI you want to use. "
                )
                time.sleep(1)
            sys.argv.insert(1, "v1")
        tp()

    if __name__ == "__main__":
        cli()
