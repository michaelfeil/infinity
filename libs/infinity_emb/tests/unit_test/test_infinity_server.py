import subprocess

import typer
import uvicorn
from fastapi import FastAPI

from infinity_emb.args import EngineArgs
from infinity_emb.infinity_server import (
    UVICORN_LOG_LEVELS,
    DeviceTypeHint,
    DtypeTypeHint,
    InferenceEngineTypeHint,
    PoolingMethodTypeHint,
    _start_uvicorn,
    cli,
    create_server,
)


def test_cli_help():
    log = subprocess.run(["infinity_emb", "--help"])
    assert log.returncode == 0


def test_patched_cli_help(mocker):
    mocker.patch("typer.run")
    cli()
    typer.run.assert_called_once_with(_start_uvicorn)


def test_cli_wrong_batch_size():
    log = subprocess.run(["infinity_emb", "--batch-size", "WrongArgument"])
    assert log.returncode == 2


def test_create_server():
    app = create_server(EngineArgs(engine="debugengine"))
    assert isinstance(app, FastAPI)


def test_patched_create_uvicorn(mocker):
    mocker.patch("uvicorn.run")
    _start_uvicorn(
        log_level=UVICORN_LOG_LEVELS.debug,  # type: ignore
        engine=InferenceEngineTypeHint.torch,
        device=DeviceTypeHint.auto,
        dtype=DtypeTypeHint.auto,
        pooling_method=PoolingMethodTypeHint.auto,
    )
    assert uvicorn.run.call_count == 1
