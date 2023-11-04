import subprocess

from fastapi import FastAPI
import uvicorn
import typer
from infinity_emb.infinity_server import create_server, start_uvicorn, InferenceEngineTypeHint, UVICORN_LOG_LEVELS, cli

from infinity_emb.transformer.utils import InferenceEngine


def test_cli_help():
    log = subprocess.run(["infinity_emb", "--help"])
    assert log.returncode == 0

def test_patched_cli_help(mocker):
    mocker.patch('typer.run')
    cli()
    typer.run.assert_called_once_with(start_uvicorn)


def test_cli_wrong_batch_size():
    log = subprocess.run(["infinity_emb", "--batch-size", "WrongArgument"])
    assert log.returncode == 2
    stub.assert_called_once_with('foo', 'bar')


def test_create_server():
    app = create_server(engine=InferenceEngine.debugengine)
    assert isinstance(app, FastAPI)

def test_patched_create_uvicorn(mocker):
    mocker.patch('uvicorn.run')
    start_uvicorn(
        log_level=UVICORN_LOG_LEVELS.debug,  # type: ignore
        engine = InferenceEngineTypeHint.torch, 
    )
    assert uvicorn.run.call_count == 1
    