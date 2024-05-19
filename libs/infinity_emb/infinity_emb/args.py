import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional

from infinity_emb._optional_imports import CHECK_PYDANTIC
from infinity_emb.primitives import (
    Device,
    Dtype,
    EmbeddingDtype,
    InferenceEngine,
    PoolingMethod,
)

if CHECK_PYDANTIC.is_available:
    from pydantic.dataclasses import dataclass as dataclass_pydantic
# if python>=3.10 use kw_only
dataclass_args = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **dataclass_args)
class EngineArgs:
    """Creating a Async EmbeddingEngine object.

    Args:
        model_name_or_path, str:  Defaults to "michaelfeil/bge-small-en-v1.5".
        batch_size, int: Defaults to 32.
        revision, str: Defaults to None.
        trust_remote_code, bool: Defaults to True.
        engine, InferenceEngine or str: backend for inference.
            Defaults to InferenceEngine.torch.
        model_warmup, bool: decide if warmup with max batch size . Defaults to True.
        vector_disk_cache_path, str: file path to folder of cache.
            Defaults to "" - default no caching.
        device, Device or str: device to use for inference. Defaults to Device.auto.
        compile, bool: compile model for better performance. Defaults to False.
        bettertransformer, bool: use bettertransformer. Defaults to True.
        dtype, Dtype or str: data type to use for inference. Defaults to Dtype.auto.
        pooling_method, PoolingMethod or str: pooling method to use. Defaults to PoolingMethod.auto.
        lengths_via_tokenize, bool: schedule by token usage. Defaults to False.
        served_model_name, str: Defaults to readable name of model_name_or_path.
    """

    model_name_or_path: str = "michaelfeil/bge-small-en-v1.5"
    batch_size: int = 32
    revision: Optional[str] = None
    trust_remote_code: bool = True
    engine: InferenceEngine = InferenceEngine.torch
    model_warmup: bool = False
    vector_disk_cache_path: str = ""
    device: Device = Device.auto
    compile: bool = not os.environ.get("INFINITY_DISABLE_COMPILE", "Disable")
    bettertransformer: bool = True
    dtype: Dtype = Dtype.auto
    pooling_method: PoolingMethod = PoolingMethod.auto
    lengths_via_tokenize: bool = False
    embedding_dtype: EmbeddingDtype = EmbeddingDtype.float32
    served_model_name: str = ""

    def __post_init__(self):
        # convert the following strings to enums
        # so they don't need to be exported to the external interface
        if not isinstance(self.engine, InferenceEngine):
            object.__setattr__(self, "engine", InferenceEngine[self.engine])
        if not isinstance(self.device, Device):
            if self.device is None:
                object.__setattr__(self, "device", Device.auto)
            else:
                object.__setattr__(self, "device", Device[self.device])
        if not isinstance(self.dtype, Dtype):
            object.__setattr__(self, "dtype", Dtype[self.dtype])
        if not isinstance(self.pooling_method, PoolingMethod):
            object.__setattr__(
                self, "pooling_method", PoolingMethod[self.pooling_method]
            )
        if not isinstance(self.embedding_dtype, EmbeddingDtype):
            object.__setattr__(
                self, "embedding_dtype", EmbeddingDtype[self.embedding_dtype]
            )
        if not self.served_model_name:
            object.__setattr__(
                self,
                "served_model_name",
                "/".join(self.model_name_or_path.split("/")[-2:]),
            )
        if self.revision is not None and self.revision == "":
            object.__setattr__(self, "revision", None)
        if isinstance(self.vector_disk_cache_path, bool):
            object.__setattr__(
                self,
                "vector_disk_cache_path",
                f"{self.engine}_{self.model_name_or_path.replace('/','_')}"
                if self.vector_disk_cache_path
                else "",
            )

        # after all done -> check if the dataclass is valid
        if CHECK_PYDANTIC.is_available:
            # convert to pydantic dataclass
            # and check if the dataclass is valid
            @dataclass_pydantic(frozen=True, **dataclass_args)
            class EngineArgsPydantic(EngineArgs):
                def __post_init__(self):
                    # overwrite the __post_init__ method
                    # to avoid infinite recursion
                    pass

            # validate
            EngineArgsPydantic(**self.__dict__)

    def to_dict(self):
        return asdict(self)
