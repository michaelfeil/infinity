# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import asyncio
import os
import signal
import time
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional, Union, TYPE_CHECKING

import infinity_emb
from infinity_emb.args import EngineArgs
from infinity_emb.engine import AsyncEmbeddingEngine, AsyncEngineArray
from infinity_emb.env import MANAGER
from infinity_emb.fastapi_schemas import docs, errors
from infinity_emb.log_handler import logger
from infinity_emb.primitives import (
    AudioCorruption,
    ImageCorruption,
    Modality,
    ModelCapabilites,
    MatryoshkaDimError,
    ModelNotDeployedError,
)
from infinity_emb.telemetry import PostHog, StartupTelemetry, telemetry_log_info

if TYPE_CHECKING:
    from infinity_emb.fastapi_schemas.pymodels import DataURIorURL


def send_telemetry_start(
    engine_args_list: list[EngineArgs],
    capabilities_list: list[set[ModelCapabilites]],
):
    time.sleep(60)
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
    from infinity_emb.fastapi_schemas.pymodels import (
        AudioEmbeddingInput,
        ClassifyInput,
        ClassifyResult,
        ImageEmbeddingInput,
        MultiModalOpenAIEmbedding,
        OpenAIEmbeddingResult,
        OpenAIModelInfo,
        RerankInput,
        ReRankResult,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        instrumentator.expose(app)  # type: ignore
        logger.info(
            f"Creating {len(engine_args_list)}engines: engines={[e.served_model_name for e in engine_args_list]}"
        )
        telemetry_log_info()
        app.engine_array = AsyncEngineArray.from_args(engine_args_list)  # type: ignore
        th = threading.Thread(
            target=send_telemetry_start,
            args=(engine_args_list, [e.capabilities for e in app.engine_array]),  # type: ignore
        )
        th.daemon = True
        th.start()
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

            logger.info(f"Preloaded configuration successfully. {engine_args_list} " " -> exit .")
            asyncio.create_task(kill_later(3))

        yield
        await app.engine_array.astop()  # type: ignore
        # shutdown!

    app = FastAPI(
        title=docs.FASTAPI_TITLE,
        summary=docs.FASTAPI_SUMMARY,
        description=docs.FASTAPI_DESCRIPTION,
        version=infinity_emb.__version__,
        contact=dict(name="Michael Feil, Raphael Wirth"),  # codespell:ignore
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
        inputs: Union["DataURIorURL", list["DataURIorURL"]],
    ) -> list[Union[str, bytes]]:
        if hasattr(inputs, "host"):
            # if it is a single url
            urls_or_bytes: list[Union[str, bytes]] = [str(inputs)]
        elif hasattr(inputs, "mimetype"):
            urls_or_bytes: list[Union[str, bytes]] = [inputs.data]  # type: ignore
        else:
            # is list, resolve to bytes or url
            urls_or_bytes: list[Union[str, bytes]] = [  # type: ignore
                str(d) if hasattr(d, "host") else d.data
                for d in inputs  # type: ignore
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
                embedding, usage = await engine.embed(
                    sentences=input_, matryoshka_dim=data_root.dimensions
                )
            elif modality == Modality.audio:
                urls_or_bytes = _resolve_mixed_input(data_root.input)  # type: ignore
                logger.debug(
                    "[📝] Received request with %s input audios ",
                    len(urls_or_bytes),  # type: ignore
                )
                embedding, usage = await engine.audio_embed(
                    audios=urls_or_bytes, matryoshka_dim=data_root.dimensions
                )
            elif modality == Modality.image:
                urls_or_bytes = _resolve_mixed_input(data_root.input)  # type: ignore
                logger.debug(
                    "[📝] Received request with %s input images ",
                    len(urls_or_bytes),  # type: ignore
                )
                embedding, usage = await engine.image_embed(
                    images=urls_or_bytes, matryoshka_dim=data_root.dimensions
                )

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
        except (ImageCorruption, AudioCorruption, MatryoshkaDimError) as ex:
            raise errors.OpenAIException(
                f"{ex.__class__} -> {ex}",
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
        except (ImageCorruption, MatryoshkaDimError) as ex:
            raise errors.OpenAIException(
                f"{ex.__class__} -> {ex}",
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
        except (AudioCorruption, MatryoshkaDimError) as ex:
            raise errors.OpenAIException(
                f"{ex.__class__} -> {ex}",
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
