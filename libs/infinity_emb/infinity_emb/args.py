import sys
from dataclasses import dataclass
from typing import Optional

from infinity_emb.primitives import Device, Dtype, InferenceEngine, PoolingMethod

# if python>=3.10 use kw_only
dataclass_args = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **dataclass_args)
class EngineArgs:
    """Creating a Async EmbeddingEngine object.

    Args:
        model_name_or_path, str:  Defaults to "michaelfeil/bge-small-en-v1.5".
        batch_size, int: Defaults to 64.
        revision, str: Defaults to None.
        trust_remote_code, bool: Defaults to True.
        engine, InferenceEngine or str: backend for inference.
            Defaults to InferenceEngine.torch.
        model_warmup, bool: decide if warmup with max batch size . Defaults to True.
        vector_disk_cache_path, str: file path to folder of cache.
            Defaults to "" - default no caching.
        device, Device or str: device to use for inference. Defaults to Device.auto,
        lengths_via_tokenize, bool: schedule by token usage. Defaults to False
    """

    model_name_or_path: str = "michaelfeil/bge-small-en-v1.5"
    batch_size: int = 64
    revision: Optional[str] = None
    trust_remote_code: bool = True
    engine: InferenceEngine = InferenceEngine.torch
    model_warmup: bool = False
    vector_disk_cache_path: str = ""
    device: Device = Device.auto
    compile: bool = False
    bettertransformer: bool = True
    dtype: Dtype = Dtype.auto
    pooling_method: PoolingMethod = PoolingMethod.auto
    lengths_via_tokenize: bool = False

    def __post_init__(self):
        # convert the following strings to enums
        # so they don't need to be exported to the external interface
        if isinstance(self.engine, str):
            object.__setattr__(self, "engine", InferenceEngine[self.engine])
        if isinstance(self.device, str):
            object.__setattr__(self, "device", Device[self.device])
        if isinstance(self.dtype, str):
            object.__setattr__(self, "dtype", Dtype[self.dtype])
