from pathlib import Path
from typing import List, Union

from infinity_emb.log_handler import logger

try:
    from huggingface_hub import HfApi, HfFolder  # type: ignore
    from huggingface_hub.constants import HF_HUB_CACHE  # type: ignore
    from optimum.onnxruntime import ORTOptimizer  # type: ignore
    from optimum.onnxruntime.configuration import OptimizationConfig  # type: ignore
except ImportError:
    pass
except RuntimeError:
    pass


def device_to_onnx(device) -> Union[str, List[str]]:
    if device == "cpu":
        return "CPUExecutionProvider"
    elif device == "cuda":
        return "CUDAExecutionProvider"
    elif device == "mps":
        return "CoreMLExecutionProvider"
    elif device == "tensorrt":
        return "TensorrtExecutionProvider"
    elif device is None or device == "auto":
        raise ValueError(
            "for the optimum engine, the `device=auto` "
            "is not supported yet, select `cpu` or `cuda` explicitly"
        )
    else:
        raise ValueError(f"Unknown device {device}")


def optimize_model(
    model_name_or_path,
    model_class,
    execution_provider: str,
    file_name: str,
    optimize_model=False,
):
    path_folder = (
        Path(model_name_or_path)
        if Path(model_name_or_path).exists()
        else Path(HF_HUB_CACHE) / "infinity_onnx" / model_name_or_path
    )
    files_optimized = list(path_folder.glob("**/*optimized.onnx"))
    if files_optimized and not execution_provider == "TensorrtExecutionProvider":
        file_optimized = files_optimized[0]
        logger.info(f"Optimized model found at {file_optimized}, skipping optimization")
        return model_class.from_pretrained(
            file_optimized.parent.as_posix(),
            provider=execution_provider,
            file_name=file_optimized.name,
        )

    unoptimized_model_path = model_class.from_pretrained(
        model_name_or_path, provider=execution_provider, file_name=file_name
    )
    if not optimize_model or execution_provider == "TensorrtExecutionProvider":
        return unoptimized_model_path

    try:
        logger.info("Optimizing model")

        optimizer = ORTOptimizer.from_pretrained(unoptimized_model_path)

        optimization_config = OptimizationConfig(
            optimization_level=99,
            optimize_with_onnxruntime_only=False,
            optimize_for_gpu="cpu" not in execution_provider.lower(),
        )

        optimized_model_path = optimizer.optimize(
            optimization_config=optimization_config,
            save_dir=path_folder.as_posix(),
            # if larger than 2gb use external data format
            one_external_file=True,
        )

        model = model_class.from_pretrained(
            optimized_model_path,
            provider=execution_provider,
            file_name=Path(file_name).name.replace(".onnx", "_optimized.onnx"),
        )
    except Exception as e:
        logger.warning(
            f"Optimization failed with {e}. Going to use the unoptimized model."
        )
        model = unoptimized_model_path

    return model


def get_onnx_files(
    model_id: str,
    revision: str,
    use_auth_token: Union[bool, str] = True,
    prefer_quantized=False,
) -> Path:
    """gets the onnx files from the repo"""
    if isinstance(use_auth_token, bool):
        token = HfFolder().get_token()
    else:
        token = use_auth_token
    repo_files = map(
        Path, HfApi().list_repo_files(model_id, revision=revision, token=token)
    )
    pattern = "**.onnx"
    onnx_files = [p for p in repo_files if p.match(pattern)]

    prefered_regex = "quantize" if prefer_quantized else "model.onnx"
    prefered_onnx = [f for f in onnx_files if prefered_regex in f.name]
    if len(onnx_files) > 1:
        logger.info(f"Found {len(onnx_files)} onnx files: {onnx_files}")
        if prefered_onnx:
            onnx_files = prefered_onnx
        onnx_file = onnx_files[-1]
        logger.info(f"Using {onnx_file} as the model")
        return onnx_file
    elif len(onnx_files) == 1:
        return onnx_files[0]
    else:
        raise ValueError(f"No onnx files found for {model_id}")
