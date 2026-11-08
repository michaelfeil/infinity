from pathlib import Path
from unittest.mock import Mock

import pytest

from infinity_emb.transformer.utils_optimum import get_onnx_files, optimize_model


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("directory", ["checkpoint", "current", "checkpoint/../current"])
def test_load_local_onnx_symlinks(tmp_path, monkeypatch, relative, directory):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "onnx").mkdir(parents=True)
    weights = tmp_path / "weights.onnx"
    weights.touch()
    try:
        (checkpoint / "onnx/model.onnx").symlink_to(weights)
        (tmp_path / "current").symlink_to(checkpoint, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise
        pytest.skip("Windows symlink privilege is unavailable")

    monkeypatch.chdir(tmp_path)
    model_path = directory if relative else str(tmp_path / directory)
    onnx_file = get_onnx_files(model_name_or_path=model_path)
    loader = Mock()

    model = optimize_model(model_path, loader, "CPUExecutionProvider", str(onnx_file))

    assert model is loader.from_pretrained.return_value
    assert loader.from_pretrained.call_args.args == (model_path,)
    kwargs = loader.from_pretrained.call_args.kwargs
    assert kwargs["file_name"] == "model.onnx"
    assert kwargs["subfolder"] == "onnx"
    assert (Path(model_path) / kwargs["subfolder"] / kwargs["file_name"]).samefile(weights)
