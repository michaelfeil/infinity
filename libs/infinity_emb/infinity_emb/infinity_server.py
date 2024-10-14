# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import asyncio
import os
import re
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

import infinity_emb
from infinity_emb._optional_imports import CHECK_TYPER, CHECK_UVICORN
from infinity_emb.args import EngineArgs
from infinity_emb.engine import AsyncEmbeddingEngine, AsyncEngineArray
from infinity_emb.env import MANAGER
from infinity_emb.fastapi_schemas import docs, errors
from infinity_emb.fastapi_schemas.pymodels import (
    AudioEmbeddingInput,
    ClassifyInput,
    ClassifyResult,
    DataURIorURL,
    ImageEmbeddingInput,
    MultiModalOpenAIEmbedding,
    OpenAIEmbeddingResult,
    OpenAIModelInfo,
    RerankInput,
    ReRankResult,
)
from infinity_emb.log_handler import UVICORN_LOG_LEVELS, logger
from infinity_emb.primitives import (
    AudioCorruption,
    Device,
    Dtype,
    EmbeddingDtype,
    ImageCorruption,
    InferenceEngine,
    Modality,
    ModelCapabilites,
    ModelNotDeployedError,
    PoolingMethod,
)
from infinity_emb.telemetry import PostHog, StartupTelemetry


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
    creates the FastAPI server for a set of EngineArgs.

    """
    from fastapi import Depends, FastAPI, HTTPException, responses, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from prometheus_fastapi_instrumentator import Instrumentator

    def send_telemetry_start(
        engine_args_list: list[EngineArgs],
        capabilities_list: list[set[ModelCapabilites]],
    ):
        session_id = uuid.uuid4().hex
        for arg, capabilities in zip(engine_args_list, capabilities_list):
            PostHog.capture(
                StartupTelemetry(
                    engine_args=arg,
                    num_engines=len(engine_args_list),
                    capabilities=capabilities,
                    session_id=session_id,
                )
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        instrumentator.expose(app)  # type: ignore
        app.engine_array = AsyncEngineArray.from_args(engine_args_list)  # type: ignore
        asyncio.create_task(
            asyncio.to_thread(
                send_telemetry_start,
                engine_args_list,
                [e.capabilities for e in app.engine_array],  # type: ignore
            )
        )
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

    def _resolve_mixed_input(
        inputs: Union[DataURIorURL, list[DataURIorURL]]
    ) -> list[Union[str, bytes]]:
        if hasattr(inputs, "host"):
            # if it is a single url
            urls_or_bytes: list[Union[str, bytes]] = [str(inputs)]
        elif hasattr(inputs, "mimetype"):
            urls_or_bytes: list[Union[str, bytes]] = [inputs]  # type: ignore
        else:
            # is list, resolve to bytes or url
            urls_or_bytes: list[Union[str, bytes]] = [  # type: ignore
                str(d) if hasattr(d, "host") else d.data for d in inputs  # type: ignore
            ]
        return urls_or_bytes

    @app.post(
        f"{url_prefix}/embeddings",
        response_model=OpenAIEmbeddingResult,
        response_class=responses.ORJSONResponse,
        dependencies=route_dependencies,
        operation_id="embeddings",
    )
    async def _embeddings(data: MultiModalOpenAIEmbedding):
        """Encode Embeddings. Supports with multimodal inputs. Aligned with OpenAI Embeddings API.

        ## Running Text Embeddings
        ```python
        import requests, base64
        requests.post("http://..:7997/embeddings",
            json={"model":"openai/clip-vit-base-patch32","input":["Two cute cats."]})
        ```

        ## Running Image Embeddings
        ```python
        requests.post("http://..:7997/embeddings",
            json={
                "model": "openai/clip-vit-base-patch32",
                "encoding_format": "base64",
                "input": [
                    "http://images.cocodataset.org/val2017/000000039769.jpg",
                    # can also be base64 encoded
                ],
                # set extra modality to image to process as image
                "modality": "image"
        )
        ```

        ## Running Audio Embeddings
        ```python
        import requests, base64
        url = "https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/libs/infinity_emb/tests/data/audio/beep.wav"

        def url_to_base64(url, modality = "image"):
            '''small helper to convert url to base64 without server requiring access to the url'''
            response = requests.get(url)
            response.raise_for_status()
            base64_encoded = base64.b64encode(response.content).decode('utf-8')
            mimetype = f"{modality}/{url.split('.')[-1]}"
            return f"data:{mimetype};base64,{base64_encoded}"

        requests.post("http://localhost:7997/embeddings",
            json={
                "model": "laion/larger_clap_general",
                "encoding_format": "float",
                "input": [
                    url, url_to_base64(url, "audio")
                ],
                # set extra modality to audio to process as audio
                "modality": "audio"
            }
        )
        ```

        ## Running via OpenAI Client
        ```python
        from openai import OpenAI # pip install openai==1.51.0
        client = OpenAI(base_url="http://localhost:7997/")
        client.embeddings.create(
            model="laion/larger_clap_general",
            input=[url_to_base64(url, "audio")],
            encoding_format="float",
            extra_body={
                "modality": "audio"
            }
        )

        client.embeddings.create(
            model="laion/larger_clap_general",
            input=["the sound of a beep", "the sound of a cat"],
            encoding_format="base64", # base64: optional high performance setting
            extra_body={
                "modality": "text"
            }
        )
        ```

        ### Hint: Run all the above models on one server:
        ```bash
        infinity_emb v2 --model-id BAAI/bge-small-en-v1.5 --model-id openai/clip-vit-base-patch32 --model-id laion/larger_clap_general
        ```
        """

        modality = data.root.modality
        data_root = data.root
        engine = _resolve_engine(data_root.model)

        try:
            start = time.perf_counter()
            if modality == Modality.text:
                if isinstance(data_root.input, str):
                    input_ = [data_root.input]
                else:
                    input_ = data_root.input  # type: ignore
                logger.debug(
                    "[📝] Received request with %s input texts ",
                    len(input_),  # type: ignore
                )
                embedding, usage = await engine.embed(sentences=input_)
            elif modality == Modality.audio:
                urls_or_bytes = _resolve_mixed_input(data_root.input)  # type: ignore
                logger.debug(
                    "[📝] Received request with %s input audios ",
                    len(urls_or_bytes),  # type: ignore
                )
                embedding, usage = await engine.audio_embed(audios=urls_or_bytes)
            elif modality == Modality.image:
                urls_or_bytes = _resolve_mixed_input(data_root.input)  # type: ignore
                logger.debug(
                    "[📝] Received request with %s input images ",
                    len(urls_or_bytes),  # type: ignore
                )
                embedding, usage = await engine.image_embed(images=urls_or_bytes)

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            return OpenAIEmbeddingResult.to_embeddings_response(
                embeddings=embedding,
                engine_args=engine.engine_args,
                encoding_format=data_root.encoding_format,
                usage=usage,
            )
        except ModelNotDeployedError as ex:
            raise errors.OpenAIException(
                f"ModelNotDeployedError: model=`{data_root.model}` does not support `embed` for modality `{modality.value}`. Reason: {ex}",
                code=status.HTTP_400_BAD_REQUEST,
            )
        except (ImageCorruption, AudioCorruption) as ex:
            # get urls_or_bytes if not defined
            try:
                urls_or_bytes = urls_or_bytes
            except NameError:
                urls_or_bytes = []
            raise errors.OpenAIException(
                f"{modality.value}Corruption, could not open {[b if isinstance(b, str) else 'bytes' for b in urls_or_bytes]} -> {ex}",
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
        """Rerank documents. Aligned with Cohere API (https://docs.cohere.com/reference/rerank)

        ```python
        import requests
        requests.post("http://..:7997/rerank",
            json={
                "model":"mixedbread-ai/mxbai-rerank-xsmall-v1",
                "query":"Where is Munich?",
                "documents":["Munich is in Germany.", "The sky is blue."]
            })
        ```
        """
        engine = _resolve_engine(data.model)
        try:
            logger.debug("[📝] Received request with %s docs ", len(data.documents))
            start = time.perf_counter()

            scores, usage = await engine.rerank(
                query=data.query,
                docs=data.documents,
                raw_scores=data.raw_scores,
                top_n=data.top_n,
            )

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            return ReRankResult.to_rerank_response(
                scores=scores,
                model=engine.engine_args.served_model_name,
                usage=usage,
                return_documents=data.return_documents,
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
        ```
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
        deprecated=True,
        summary="Deprecated: Use `embeddings` with `modality` set to `image`",
    )
    async def _embeddings_image(data: ImageEmbeddingInput):
        """Encode Embeddings from Image files

        Supports URLs of Images and Base64-encoded Images

        ```python
        import requests
        requests.post("http://..:7997/embeddings_image",
            json={
                "model":"openai/clip-vit-base-patch32",
                "input": [
                    "http://images.cocodataset.org/val2017/000000039769.jpg",
                    "data:image/png;base64,iVBORw0KGgoDEMOoSAMPLEoENCODEDIMAGE"
                ]
            })
        ```
        """
        engine = _resolve_engine(data.model)
        urls_or_bytes = _resolve_mixed_input(data.input)  # type: ignore
        try:
            logger.debug("[📝] Received request with %s Urls ", len(urls_or_bytes))
            start = time.perf_counter()

            embedding, usage = await engine.image_embed(images=urls_or_bytes)

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            return OpenAIEmbeddingResult.to_embeddings_response(
                embeddings=embedding,
                engine_args=engine.engine_args,
                encoding_format=data.encoding_format,
                usage=usage,
            )
        except ImageCorruption as ex:
            raise errors.OpenAIException(
                f"ImageCorruption, could not open {[b if isinstance(b, str) else 'bytes' for b in urls_or_bytes]} -> {ex}",
                code=status.HTTP_400_BAD_REQUEST,
            )
        except ModelNotDeployedError as ex:
            raise errors.OpenAIException(
                f"ModelNotDeployedError: model=`{data.model}` does not support `image_embed`. Reason: {ex}",
                code=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as ex:
            raise errors.OpenAIException(
                f"InternalServerError: {ex}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @app.post(
        f"{url_prefix}/embeddings_audio",
        response_model=OpenAIEmbeddingResult,
        response_class=responses.ORJSONResponse,
        dependencies=route_dependencies,
        operation_id="embeddings_audio",
        deprecated=True,
        summary="Deprecated: Use `embeddings` with `modality` set to `audio`",
    )
    async def _embeddings_audio(data: AudioEmbeddingInput):
        """Encode Embeddings from Audio files

        Supports URLs of Audios and Base64-encoded Audios

        ```python
        import requests
        requests.post("http://..:7997/embeddings_audio",
            json={
                "model":"laion/larger_clap_general",
                "input": [
                    "https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/libs/infinity_emb/tests/data/audio/beep.wav",
                    "data:audio/wav;base64,iVBORw0KGgoDEMOoSAMPLEoENCODEDAUDIO"
                ]
            })
        ```
        """
        engine = _resolve_engine(data.model)
        urls_or_bytes = _resolve_mixed_input(data.input)  # type: ignore
        try:
            logger.debug("[📝] Received request with %s Urls ", len(urls_or_bytes))
            start = time.perf_counter()

            embedding, usage = await engine.audio_embed(audios=urls_or_bytes)  # type: ignore

            duration = (time.perf_counter() - start) * 1000
            logger.debug("[✅] Done in %s ms", duration)

            return OpenAIEmbeddingResult.to_embeddings_response(
                embeddings=embedding,
                engine_args=engine.engine_args,
                encoding_format=data.encoding_format,
                usage=usage,
            )
        except AudioCorruption as ex:
            raise errors.OpenAIException(
                f"AudioCorruption, could not open {[b if isinstance(b, str) else 'bytes' for b in urls_or_bytes]} -> {ex}",
                code=status.HTTP_400_BAD_REQUEST,
            )
        except ModelNotDeployedError as ex:
            raise errors.OpenAIException(
                f"ModelNotDeployedError: model=`{data.model}` does not support `audio_embed`. Reason: {ex}",
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

    def _resolve(self, x, iteration: int):
        """pad x to length of self.length"""
        x = typer_option_resolve(x)
        if not isinstance(x, (list, tuple)):
            return x
        elif len(x) == 1:
            return x[0]
        elif len(x) == self.length:
            return x[iteration]
        else:
            raise ValueError(f"Expected length {self.length} but got {len(x)}")

    def __iter__(self):
        """iterate over kwargs and pad them to length of self.length"""
        for iteration in range(self.length):
            kwargs = {}
            for key, value in self.kwargs.items():
                kwargs[key] = self._resolve(value, iteration)
            yield kwargs


def typer_option_resolve(*args):
    """returns the value or the default value"""
    if len(args) == 1:
        return (
            args[0].default
            if hasattr(args[0], "default") and hasattr(args[0], "envvar")
            else args[0]
        )
    return (
        a.default if (hasattr(a, "default") and hasattr(a, "envvar")) else a
        for a in args
    )


# CLI
if CHECK_TYPER.is_available:
    CHECK_TYPER.mark_required()
    CHECK_UVICORN.mark_required()
    import typer
    import uvicorn

    tp = typer.Typer()

    @tp.command("v1")
    def v1(
        # v1 is deprecated. Please do no longer modify it.
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
        embedding_dtype: EmbeddingDtype = EmbeddingDtype.default_value(),  # type: ignore
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
        """Infinity API ♾️  cli v1 - deprecated, consider use cli v2 via `infinity_emb v2`."""
        if api_key:
            raise ValueError(
                "api_key is not supported in `v1`. Please migrate to `v2`."
            )
        if not (
            embedding_dtype == EmbeddingDtype.float32
            or embedding_dtype == EmbeddingDtype.default_value()
        ):
            raise ValueError(
                "selecting embedding_dtype is not supported in `v1`. Please migrate to `v2`."
            )
        logger.warning(
            "CLI v1 is deprecated. Consider use CLI `v2`, by specifying `v2` as the command."
        )
        time.sleep(1)
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
            embedding_dtype=[EmbeddingDtype.float32],
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

    def _construct(name: str):
        """constructs the default entry and type hint for the variable name"""
        return dict(
            # gets the default value from the ENV Manager
            default=getattr(MANAGER, name),
            # envvar is a dummy that is there for documentation purposes.
            envvar=f"`{MANAGER.to_name(name)}`",
        )

    def validate_url(path: str):
        """
        This regex matches:
        - An empty string or A single '/'
        - A string that starts with '/' and does not end with '/'
        """
        if re.match(r"^$|^/$|^/.*[^/]$", path):
            return path
        raise typer.BadParameter("Path must start with '/' and must not end with '/'")

    @tp.command("v2")
    def v2(
        # t
        # arguments for engine
        model_id: list[str] = typer.Option(
            **_construct("model_id"),
            help="Huggingface model repo id. Subset of possible models: https://huggingface.co/models?other=text-embeddings-inference&",
        ),
        served_model_name: list[str] = typer.Option(
            **_construct("served_model_name"),
            help="the nickname for the API, under which the model_id can be selected",
        ),
        batch_size: list[int] = typer.Option(
            **_construct("batch_size"), help="maximum batch size for inference"
        ),
        revision: list[str] = typer.Option(
            **_construct("revision"), help="huggingface  model repo revision."
        ),
        trust_remote_code: list[bool] = typer.Option(
            **_construct("trust_remote_code"),
            help="if potential remote modeling code from huggingface repo is trusted.",
        ),
        engine: list[InferenceEngine] = typer.Option(
            **_construct("engine"),
            help="Which backend to use. `torch` uses Pytorch GPU/CPU, optimum uses ONNX on GPU/CPU/NVIDIA-TensorRT, `CTranslate2` uses torch+ctranslate2 on CPU/GPU.",
        ),
        model_warmup: list[bool] = typer.Option(
            **_construct("model_warmup"),
            help="if model should be warmed up after startup, and before ready.",
        ),
        vector_disk_cache: list[bool] = typer.Option(
            **_construct("vector_disk_cache"),
            help="If hash(request)/results should be cached to SQLite for latency improvement.",
        ),
        device: list[Device] = typer.Option(
            **_construct("device"),
            help="device to use for computing the model forward pass.",
        ),
        lengths_via_tokenize: list[bool] = typer.Option(
            **_construct("lengths_via_tokenize"),
            help="if True, returned tokens is based on actual tokenizer count. If false, uses len(input) as proxy.",
        ),
        dtype: list[Dtype] = typer.Option(
            **_construct("dtype"), help="dtype for the model weights."
        ),
        embedding_dtype: list[EmbeddingDtype] = typer.Option(
            **_construct("embedding_dtype"),
            help="dtype post-forward pass. If != `float32`, using Post-Forward Static quantization.",
        ),
        pooling_method: list[PoolingMethod] = typer.Option(
            **_construct("pooling_method"),
            help="overwrite the pooling method if inferred incorrectly.",
        ),
        compile: list[bool] = typer.Option(
            **_construct("compile"),
            help="Enable usage of `torch.compile(dynamic=True)` if engine relies on it.",
        ),
        bettertransformer: list[bool] = typer.Option(
            **_construct("bettertransformer"),
            help="Enables varlen flash-attention-2 via the `BetterTransformer` implementation. If available for this model.",
        ),
        # arguments for uvicorn / server
        preload_only: bool = typer.Option(
            **_construct("preload_only"),
            help="If true, only downloads models and verifies setup, then exit. Recommended for pre-caching the download in a Dockerfile.",
        ),
        host: str = typer.Option(
            **_construct("host"), help="host for the FastAPI uvicorn server"
        ),
        port: int = typer.Option(
            **_construct("port"), help="port for the FastAPI uvicorn server"
        ),
        url_prefix: str = typer.Option(
            **_construct("url_prefix"),
            callback=validate_url,
            help="prefix for all routes of the FastAPI uvicorn server. Useful if you run behind a proxy / cascaded API.",
        ),
        redirect_slash: str = typer.Option(
            **_construct("redirect_slash"), help="where to redirect `/` requests to."
        ),
        log_level: UVICORN_LOG_LEVELS = typer.Option(**_construct("log_level"), help="console log level."),  # type: ignore
        permissive_cors: bool = typer.Option(
            **_construct("permissive_cors"), help="whether to allow permissive cors."
        ),
        api_key: str = typer.Option(
            **_construct("api_key"), help="api_key used for authentication headers."
        ),
        proxy_root_path: str = typer.Option(
            **_construct("proxy_root_path"),
            help="Proxy prefix for the application. See: https://fastapi.tiangolo.com/advanced/behind-a-proxy/",
        ),
    ):
        """Infinity API ♾️  cli v2. MIT License. Copyright (c) 2023-now Michael Feil \n
        \n
        Multiple Model CLI Playbook: \n
        - 1. cli options can be overloaded i.e. `v2 --model-id model/id1 --model-id/id2 --batch-size 8 --batch-size 4` \n
        - 2. or adapt the defaults by setting ENV Variables separated by `;`: INFINITY_MODEL_ID="model/id1;model/id2;" && INFINITY_BATCH_SIZE="8;4;" \n
        - 3. single items are broadcasted to `--model-id` length, making `v2 --model-id model/id1 --model-id/id2 --batch-size 8` both models have batch-size 8. \n
        """
        # old
        """
        model_id, list[str]: Huggingface model, e.g.
            ["michaelfeil/bge-small-en-v1.5", "mixedbread-ai/mxbai-embed-large-v1"]
            Defaults to `INFINITY_MODEL_ID`
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

        (
            url_prefix,
            host,
            port,
            redirect_slash,
            log_level,
            preload_only,
            permissive_cors,
            api_key,
            proxy_root_path,
        ) = typer_option_resolve(
            url_prefix,
            host,
            port,
            redirect_slash,
            log_level,
            preload_only,
            permissive_cors,
            api_key,
            proxy_root_path,
        )

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
        CHECK_TYPER.mark_required()
        if len(sys.argv) == 1 or sys.argv[1] not in ["v1", "v2", "help", "--help"]:
            for _ in range(3):
                logger.error(
                    "Error: No command given. Defaulting to `v1`. "
                    "Relying on this side effect is considered an error and "
                    "will be deprecated in the future, which requires explicit usage of a `infinity_emb v1` or `infinity_emb v2`. "
                    "Specify the version of the CLI you want to use. "
                )
                time.sleep(1)
            sys.argv.insert(1, "v1")
        tp()

    if __name__ == "__main__":
        cli()
