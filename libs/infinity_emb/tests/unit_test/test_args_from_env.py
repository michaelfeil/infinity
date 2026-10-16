import pytest

import infinity_emb.args as args_module
import infinity_emb.env as env_module
from infinity_emb import AsyncEngineArray, EngineArgs
from infinity_emb.cli import AutoPadding


@pytest.fixture
def fresh_env(monkeypatch):
    def configure(count, **options):
        for name in vars(env_module.MANAGER):
            if not name.startswith("_"):
                monkeypatch.delenv(f"INFINITY_{name.upper()}", raising=False)
        monkeypatch.setenv("INFINITY_MODEL_ID", ";".join(f"local/model-{i}" for i in range(count)))
        monkeypatch.setenv("INFINITY_ENGINE", "debugengine")
        monkeypatch.setenv("INFINITY_DEVICE", "cpu")
        for name, value in options.items():
            monkeypatch.setenv(f"INFINITY_{name.upper()}", value)
        manager = type(env_module.MANAGER)()
        monkeypatch.setattr(env_module, "MANAGER", manager)
        monkeypatch.setattr(args_module, "MANAGER", manager)
        return manager

    return configure


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("mode", ["defaults", "singleton", "per_model"])
def test_from_env_model_options(fresh_env, count, mode):
    options = {}
    if mode == "singleton":
        options = dict(
            batch_size="8",
            model_warmup="false",
            revision="revision-one",
            dtype="float32",
            onnx_disable_optimize="true",
        )
    elif mode == "per_model":
        options = dict(
            batch_size=";".join(str(8 + i) for i in range(count)),
            model_warmup=";".join("true" if i % 2 else "false" for i in range(count)),
            revision=";".join(f"revision-{i}" for i in range(count)),
            dtype=";".join("float32" for _ in range(count)),
            onnx_disable_optimize=";".join("true" for _ in range(count)),
        )
    fresh_env(count, **options)
    args = EngineArgs.from_env()
    assert len(args) == count
    for i, arg in enumerate(args):
        assert arg.model_name_or_path == f"local/model-{i}"
        assert arg.served_model_name == f"local/model-{i}"
        assert arg.engine.value == "debugengine"
        assert arg.device.value == "cpu"
        assert arg.batch_size == (
            32 if mode == "defaults" else 8 + (i if mode == "per_model" else 0)
        )
        assert arg.model_warmup == (
            True if mode == "defaults" else bool(i % 2) if mode == "per_model" else False
        )
        assert arg.revision == (
            None
            if mode == "defaults"
            else f"revision-{i}"
            if mode == "per_model"
            else "revision-one"
        )
        assert arg.dtype.value == ("auto" if mode == "defaults" else "float32")
        assert arg.onnx_disable_optimize == (mode != "defaults")


@pytest.mark.parametrize("count, values", [(1, [8, 16]), (2, [8, 16, 32]), (3, [8, 16])])
def test_from_env_mismatched_lengths_match_cli(fresh_env, count, values):
    fresh_env(count, batch_size=";".join(map(str, values)))
    message = f"Expected length {count} but got {len(values)}"
    with pytest.raises(ValueError, match=message):
        list(AutoPadding(length=count, batch_size=values))
    with pytest.raises(ValueError, match=message):
        EngineArgs.from_env()


@pytest.mark.anyio
@pytest.mark.parametrize("count", [1, 2, 3])
async def test_from_env_native_debug_lifecycle(fresh_env, count):
    fresh_env(count, model_warmup="false", batch_size="8")
    array = AsyncEngineArray.from_args(EngineArgs.from_env())
    try:
        await array.astart()
        for engine in array:
            embeddings, usage = await engine.embed(["abc", "hello"])
            assert [embedding[0] for embedding in embeddings] == [3, 5]
            assert usage == 8
    finally:
        await array.astop()
    assert not array.is_running()


@pytest.mark.parametrize("count", [2, 3])
def test_from_env_fully_explicit_parallel_options(fresh_env, count):
    options = {
        "batch_size": "8",
        "revision": "main",
        "trust_remote_code": "false",
        "engine": "debugengine",
        "model_warmup": "false",
        "device": "cpu",
        "compile": "false",
        "bettertransformer": "false",
        "dtype": "float32",
        "pooling_method": "mean",
        "lengths_via_tokenize": "false",
        "embedding_dtype": "float32",
        "onnx_disable_optimize": "true",
        "onnx_do_not_prefer_quantized": "true",
    }
    options = {key: ";".join([value] * count) for key, value in options.items()}
    options["served_model_name"] = ";".join(f"model-{i}" for i in range(count))
    fresh_env(count, **options)
    args = EngineArgs.from_env()
    assert len(args) == count
    assert [arg.served_model_name for arg in args] == [f"model-{i}" for i in range(count)]
    assert all(arg.batch_size == 8 and arg.revision == "main" for arg in args)
    assert all(not arg.trust_remote_code and not arg.model_warmup for arg in args)
    assert all(arg.onnx_disable_optimize and arg.onnx_do_not_prefer_quantized for arg in args)


def test_from_env_trailing_separator(fresh_env):
    fresh_env(2, model_id="local/model-0;local/model-1;", batch_size="8;16;")
    args = EngineArgs.from_env()
    assert [arg.model_name_or_path for arg in args] == ["local/model-0", "local/model-1"]
    assert [arg.batch_size for arg in args] == [8, 16]


def test_cli_config_helpers_remain_importable():
    from infinity_emb import _config, cli

    assert cli.AutoPadding is _config.AutoPadding
    assert cli.typer_option_resolve is _config.typer_option_resolve
