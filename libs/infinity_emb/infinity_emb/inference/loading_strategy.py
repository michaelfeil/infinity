from infinity_emb._optional_imports import CHECK_TORCH, CHECK_TRANSFORMERS
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import InferenceEngine, Device, Dtype, DeviceID, LoadingStrategy


if CHECK_TORCH.is_available:
    import torch
if CHECK_TRANSFORMERS.is_available:
    from transformers import is_torch_npu_available  # type: ignore


def _validate_availale_device_ids(
    device: str, available_devices: list[int], desired_device_ids: DeviceID
) -> list[str]:
    if desired_device_ids:
        for device_id in desired_device_ids:
            if device_id not in available_devices:
                available_devices_str = ",".join(f"{device}:{i}" for i in available_devices)
                raise ValueError(
                    f"Device {device}:{device_id} is not available. Available devices: {available_devices_str}"
                )
        used_ids = list(desired_device_ids)
    else:
        used_ids = available_devices
    return [f"{device}:{device_id}" for device_id in used_ids]


def get_loading_strategy_torch(args: EngineArgs) -> LoadingStrategy:
    CHECK_TORCH.mark_required()

    if args.device == Device.auto:
        if torch.cuda.is_available():
            autodevice = "cuda"
        elif is_torch_npu_available():
            autodevice = "npu"
        elif torch.backends.mps.is_available():
            autodevice = "mps"
        else:
            autodevice = "cpu"
    else:
        autodevice = args.device.value

    # mix with device_id
    if autodevice == "cuda":
        autodevice_string = _validate_availale_device_ids(
            "cuda", list(range(torch.cuda.device_count())), args.device_id
        )
    elif autodevice == "npu":
        autodevice_string = _validate_availale_device_ids(
            "npu",
            list(range(torch.npu.device_count())),  # type: ignore
            args.device_id,
        )
    elif autodevice == "mps":
        autodevice_string = _validate_availale_device_ids(
            "mps", list(range(torch.mps.device_count())), args.device_id
        )
    elif autodevice == "cpu":
        # spawn multiple processes on CPU. This is useful for debugging, but not for performance.
        autodevice_string = ["cpu"] * max(len(args.device_id), 1)
    else:
        raise ValueError(f"Unknown device {autodevice}")

    # automatic dtype
    if args.dtype == Dtype.float32:
        autodtype = torch.float32
    elif args.dtype == Dtype.float16:
        autodtype = torch.float16
    elif args.dtype == Dtype.bfloat16:
        autodtype = torch.bfloat16
    elif args.dtype == Dtype.auto:
        if autodevice == "cuda":
            if torch.cuda.is_bf16_supported():
                autodtype = torch.bfloat16
            else:
                autodtype = torch.float16
        else:
            autodtype = torch.float32
    else:
        autodtype = torch.float32  # TODO: Lazy loading

    # mix with dtype
    loading_dtype = autodtype

    # mix with quantization dtype
    if args.dtype == Dtype.int8:
        quantization_dtype = torch.int8
    elif args.dtype == Dtype.fp8:
        quantization_dtype = torch.float8_e5m2
    elif loading_dtype in [torch.float32, torch.bfloat16, torch.float16]:
        quantization_dtype = None
    else:
        raise ValueError(f"Unknown dtype {args.dtype}")

    return LoadingStrategy(
        device_mapping=autodevice_string,
        loading_dtype=loading_dtype,
        quantization_dtype=quantization_dtype,
        device_placement=autodevice_string[0],
    )


def get_loading_strategy(args: EngineArgs) -> LoadingStrategy:
    if args.engine in [InferenceEngine.torch, InferenceEngine.ctranslate2]:
        stat = get_loading_strategy_torch(args)
    else:
        stat = LoadingStrategy(
            device_mapping=["not-specified"], loading_dtype=None, quantization_dtype=None
        )

    return stat
