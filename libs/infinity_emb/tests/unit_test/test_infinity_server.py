import subprocess

import numpy as np
import pytest
import typer
import uvicorn
from fastapi import FastAPI

from infinity_emb import AsyncEmbeddingEngine, transformer
from infinity_emb.infinity_server import (
    UVICORN_LOG_LEVELS,
    InferenceEngineTypeHint,
    cli,
    create_server,
    start_uvicorn,
)
from infinity_emb.transformer.utils import InferenceEngine


@pytest.mark.anyio
async def test_async_api_debug():
    sentences = ["Embedded this is sentence via Infinity.", "Paris is in France."]
    engine = AsyncEmbeddingEngine(engine=transformer.InferenceEngine.debugengine)
    async with engine:
        embeddings = np.array(await engine.embed(sentences))
        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] >= 10
        for idx, s in enumerate(sentences):
            assert embeddings[idx][0] == len(s), f"{embeddings}, {s}"


@pytest.mark.anyio
async def test_async_api_torch():
    sentences = ["Hi", "how"]
    engine = AsyncEmbeddingEngine(engine=transformer.InferenceEngine.torch)
    async with engine:
        embeddings = np.array(await engine.embed(sentences))
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] >= 10


@pytest.mark.anyio
async def test_async_api_fastembed():
    sentences = ["Hi", "how"]
    engine = AsyncEmbeddingEngine(engine=transformer.InferenceEngine.fastembed)
    async with engine:
        embeddings = np.array(await engine.embed(sentences))
        assert not engine.is_overloaded()
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] >= 10


@pytest.mark.anyio
async def test_async_api_failing():
    sentences = ["Hi", "how"]
    engine = AsyncEmbeddingEngine()
    with pytest.raises(ValueError):
        await engine.embed(sentences)

    await engine.astart()
    assert not engine.is_overloaded()
    assert engine.overload_status()

    with pytest.raises(ValueError):
        await engine.astart()
    await engine.astop()


def test_cli_help():
    log = subprocess.run(["infinity_emb", "--help"])
    assert log.returncode == 0


def test_patched_cli_help(mocker):
    mocker.patch("typer.run")
    cli()
    typer.run.assert_called_once_with(start_uvicorn)


def test_cli_wrong_batch_size():
    log = subprocess.run(["infinity_emb", "--batch-size", "WrongArgument"])
    assert log.returncode == 2


def test_create_server():
    app = create_server(engine=InferenceEngine.debugengine)
    assert isinstance(app, FastAPI)


def test_patched_create_uvicorn(mocker):
    mocker.patch("uvicorn.run")
    start_uvicorn(
        log_level=UVICORN_LOG_LEVELS.debug,  # type: ignore
        engine=InferenceEngineTypeHint.torch,
    )
    assert uvicorn.run.call_count == 1
