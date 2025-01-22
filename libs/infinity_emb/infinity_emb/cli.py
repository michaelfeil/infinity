# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import asyncio
import re
import sys


import infinity_emb
from infinity_emb._optional_imports import CHECK_TYPER, CHECK_UVICORN
from infinity_emb.args import EngineArgs
from infinity_emb.env import MANAGER
from infinity_emb.log_handler import UVICORN_LOG_LEVELS, logger
from infinity_emb.primitives import (
    Device,
    DeviceID,
    Dtype,
    EmbeddingDtype,
    InferenceEngine,
    PoolingMethod,
)
from infinity_emb.infinity_server import create_server


# helper functions for the CLI


def validate_url(path: str):
    """
    This regex matches:
    - An empty string or A single '/'
    - A string that starts with '/' and does not end with '/'
    """
    if re.match(r"^$|^/$|^/.*[^/]$", path):
        return path
    raise typer.BadParameter("Path must start with '/' and must not end with '/'")


class AutoPadding:
    """itertools.cycle with custom behaviour to pad to max length"""

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
            args[0].default  # if it is a typer option
            if hasattr(args[0], "default") and hasattr(args[0], "envvar")
            else args[0]  # if it is a normal value
        )
    return (a.default if (hasattr(a, "default") and hasattr(a, "envvar")) else a for a in args)


def _construct(name: str):
    """constructs the default entry and type hint for the variable name"""
    return dict(
        # gets the default value from the ENV Manager
        default=getattr(MANAGER, name),
        # envvar is a dummy that is there for documentation purposes.
        envvar=f"`{MANAGER.to_name(name)}`",
    )


# CLI
if CHECK_TYPER.is_available:
    CHECK_TYPER.mark_required()
    CHECK_UVICORN.mark_required()
    import typer
    import uvicorn

    # patch the asyncio scheduler with uvloop
    # which has theoretical speed-ups vs asyncio
    loopname = "auto"
    if sys.version_info < (3, 12):
        try:
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            loopname = "uvloop"
        except ImportError:
            # Windows does not support uvloop
            pass

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
        engine: "InferenceEngine" = MANAGER.engine[0],  # type: ignore # noqa
        model_warmup: bool = MANAGER.model_warmup[0],
        vector_disk_cache: bool = MANAGER.vector_disk_cache[0],
        device: "Device" = MANAGER.device[0],  # type: ignore
        lengths_via_tokenize: bool = MANAGER.lengths_via_tokenize[0],
        dtype: Dtype = MANAGER.dtype[0],  # type: ignore
        embedding_dtype: "EmbeddingDtype" = EmbeddingDtype.default_value(),  # type: ignore
        pooling_method: "PoolingMethod" = MANAGER.pooling_method[0],  # type: ignore
        compile: bool = MANAGER.compile[0],
        bettertransformer: bool = MANAGER.bettertransformer[0],
        preload_only: bool = MANAGER.preload_only,
        permissive_cors: bool = MANAGER.permissive_cors,
        api_key: str = MANAGER.api_key,
        url_prefix: str = MANAGER.url_prefix,
        host: str = MANAGER.host,
        port: int = MANAGER.port,
        log_level: "UVICORN_LOG_LEVELS" = MANAGER.log_level,  # type: ignore
    ):
        """Infinity API ♾️  cli v1 - deprecated, consider use cli v2 via `infinity_emb v2`."""
        if api_key:
            # encourage switch to v2
            raise ValueError("api_key is not supported in `v1`. Please migrate to `v2`.")
        if not (
            embedding_dtype == EmbeddingDtype.float32
            or embedding_dtype == EmbeddingDtype.default_value()
        ):
            # encourage switch to v2
            raise ValueError(
                "selecting embedding_dtype is not supported in `v1`. Please migrate to `v2`."
            )
        logger.warning(
            "CLI v1 is deprecated. Consider use CLI `v2`, by specifying `v2` as the command."
        )
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
            embedding_dtype=[EmbeddingDtype.float32],  # set to float32
            # unique kwargs
            preload_only=preload_only,
            url_prefix=url_prefix,
            host=host,
            port=port,
            redirect_slash=redirect_slash,
            log_level=log_level,
            permissive_cors=permissive_cors,
            api_key=api_key,
            proxy_root_path="",  # set as empty string
        )

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
        device_id: list[str] = typer.Option(
            **_construct("device_id"),
            help="device id defines the model placement. e.g. `0,1` will place the model on MPS/CUDA/GPU 0 and 1 each",
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
        host: str = typer.Option(**_construct("host"), help="host for the FastAPI uvicorn server"),
        port: int = typer.Option(**_construct("port"), help="port for the FastAPI uvicorn server"),
        url_prefix: str = typer.Option(
            **_construct("url_prefix"),
            callback=validate_url,
            help="prefix for all routes of the FastAPI uvicorn server. Useful if you run behind a proxy / cascaded API.",
        ),
        redirect_slash: str = typer.Option(
            **_construct("redirect_slash"), help="where to redirect `/` requests to."
        ),
        log_level: "UVICORN_LOG_LEVELS" = typer.Option(
            **_construct("log_level"), help="console log level."
        ),  # type: ignore
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
        - 1. cli options can be overloaded i.e. `v2 --model-id model/id1 --model-id model/id2 --batch-size 8 --batch-size 4` \n
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
        device_id_typed = [DeviceID(d) for d in typer_option_resolve(device_id)]
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
            device_id=device_id_typed,
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

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level.name,
            http="httptools",
            loop=loopname,  # type: ignore
        )


def cli():
    CHECK_TYPER.mark_required()
    if len(sys.argv) == 1 or sys.argv[1] not in [
        "v1",
        "v2",
        "help",
        "--help",
        "--show-completion",
        "--install-completion",
    ]:
        if len(sys.argv) == 1 or sys.argv[1] not in ["v1", "v2", "help", "--help"]:
            logger.error(
                "Error: No command given. Please use infinity with the `v2` command. "
                f"This is deprecated since 0.0.32. You are on {infinity_emb.__version__}. "
                "Usage: `infinity_emb v2 --model-id BAAI/bge-large-en-v1.5. "
                "defaulting to `v2` since 0.0.75. Please pin your revision for future upgrades."
            )

            sys.argv.insert(1, "v2")
    tp()


if __name__ == "__main__":
    if "cli" in locals():
        cli()
