from pathlib import Path
from typing import List, Union

from infinity_emb.log_handler import logger

try:
    from huggingface_hub import HfApi, HfFolder  # type: ignore
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE  # type: ignore
    from optimum.onnxruntime import ORTOptimizer  # type: ignore
    from optimum.onnxruntime.configuration import OptimizationConfig  # type: ignore
except ImportError:
    pass
except RuntimeError:
    pass

try:
    import torch
except ImportError:
    torch = None  # type: ignore


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
        if torch is not None and torch.cuda.is_available():
            return "CUDAExecutionProvider"
        else:
            return "CPUExecutionProvider"
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
        else Path(HUGGINGFACE_HUB_CACHE) / "infinity_onnx" / model_name_or_path
    )
    files_optimized = list(path_folder.glob("**/*optimized.onnx"))
    if execution_provider == "TensorrtExecutionProvider":
        return model_class.from_pretrained(
            model_name_or_path,
            provider=execution_provider,
            file_name=file_name,
            provider_options={
                "trt_fp16_enable": True,
                "trt_layer_norm_fp32_fallback": True,
                "trt_cuda_graph_enable": True,  # helps small layers
                "trt_builder_optimization_level": 3,  # select between 3-5
                # int8, not working, needs calibration table.
                # "trt_int8_use_native_calibration_table": True,
                # "trt_int8_enable": "quantize" in file_name,
            },
        )
    if files_optimized:
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

        is_gpu = "cpu" not in execution_provider.lower()
        optimization_config = OptimizationConfig(
            optimization_level=99,
            optimize_with_onnxruntime_only=False,
            optimize_for_gpu=is_gpu,
            fp16=is_gpu,
            # enable_gelu_approximation=True,
            # enable_gemm_fast_gelu_fusion=True, # might not work
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


def list_all_repo_files(
    model_name_or_path: str,
    revision: str,
    use_auth_token: Union[bool, str] = True,
):
    if not Path(model_name_or_path).exists():
        if isinstance(use_auth_token, bool):
            token = HfFolder().get_token()
        else:
            token = use_auth_token
        return list(
            map(
                Path,
                HfApi().list_repo_files(
                    model_name_or_path, revision=revision, token=token
                ),
            )
        )
    else:
        return list(Path(model_name_or_path).glob("**/*"))


def get_onnx_files(
    model_name_or_path: str,
    revision: str,
    use_auth_token: Union[bool, str] = True,
    prefer_quantized=False,
) -> Path:
    """gets the onnx files from the repo"""
    repo_files = list_all_repo_files(
        model_name_or_path=model_name_or_path,
        revision=revision,
        use_auth_token=use_auth_token,
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
        raise ValueError(f"No onnx files found for {model_name_or_path}")
