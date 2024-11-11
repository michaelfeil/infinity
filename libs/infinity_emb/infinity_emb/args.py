# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import sys
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from typing import Optional
from copy import deepcopy


from infinity_emb._optional_imports import CHECK_PYDANTIC
from infinity_emb.env import MANAGER
from infinity_emb.primitives import (
    Device,
    DeviceID,
    Dtype,
    EmbeddingDtype,
    InferenceEngine,
    PoolingMethod,
    LoadingStrategy,
)

if CHECK_PYDANTIC.is_available:
    from pydantic.dataclasses import dataclass as dataclass_pydantic
    from pydantic import ConfigDict
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
        device_id, DeviceID or str: device index to use for inference.
            Defaults to [], no preferred placement.
        compile, bool: compile model for better performance. Defaults to False.
        bettertransformer, bool: use bettertransformer. Defaults to True.
        dtype, Dtype or str: data type to use for inference. Defaults to Dtype.auto.
        pooling_method, PoolingMethod or str: pooling method to use. Defaults to PoolingMethod.auto.
        lengths_via_tokenize, bool: schedule by token usage. Defaults to False.
        served_model_name, str: Defaults to readable name of model_name_or_path.
    """

    model_name_or_path: str = MANAGER.model_id[0]
    batch_size: int = MANAGER.batch_size[0]
    revision: Optional[str] = MANAGER.revision[0]
    trust_remote_code: bool = MANAGER.trust_remote_code[0]
    engine: InferenceEngine = InferenceEngine[MANAGER.engine[0]]
    model_warmup: bool = MANAGER.model_warmup[0]
    vector_disk_cache_path: str = ""
    device: Device = Device[MANAGER.device[0]]
    device_id: DeviceID = field(default_factory=lambda: DeviceID(MANAGER.device_id[0]))
    compile: bool = MANAGER.compile[0]
    bettertransformer: bool = MANAGER.bettertransformer[0]
    dtype: Dtype = Dtype[MANAGER.dtype[0]]
    pooling_method: PoolingMethod = PoolingMethod[MANAGER.pooling_method[0]]
    lengths_via_tokenize: bool = MANAGER.lengths_via_tokenize[0]
    embedding_dtype: EmbeddingDtype = EmbeddingDtype[MANAGER.embedding_dtype[0]]
    served_model_name: str = MANAGER.served_model_name[0]

    _loading_strategy: Optional[LoadingStrategy] = None

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
        if not isinstance(self.device_id, DeviceID):
            object.__setattr__(self, "device_id", DeviceID(self.device_id))
        if not isinstance(self.dtype, Dtype):
            object.__setattr__(self, "dtype", Dtype[self.dtype])
        if not isinstance(self.pooling_method, PoolingMethod):
            object.__setattr__(self, "pooling_method", PoolingMethod[self.pooling_method])
        if not isinstance(self.embedding_dtype, EmbeddingDtype):
            object.__setattr__(self, "embedding_dtype", EmbeddingDtype[self.embedding_dtype])
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
                (
                    f"{self.engine}_{self.model_name_or_path.replace('/','_')}"
                    if self.vector_disk_cache_path
                    else ""
                ),
            )

        # after all done -> check if the dataclass is valid
        if CHECK_PYDANTIC.is_available:
            # convert to pydantic dataclass
            # and check if the dataclass is valid
            @dataclass_pydantic(
                frozen=True, config=ConfigDict(arbitrary_types_allowed=True), **dataclass_args
            )
            class EngineArgsPydantic(EngineArgs):
                def __post_init__(self):
                    # overwrite the __post_init__ method
                    # to avoid infinite recursion
                    pass

            # validate
            EngineArgsPydantic(**self.__dict__)
        if self._loading_strategy is None:
            self.update_loading_strategy()
        elif isinstance(self._loading_strategy, dict):
            object.__setattr__(self, "_loading_strategy", LoadingStrategy(**self._loading_strategy))

    def to_dict(self):
        return asdict(self)

    def update_loading_strategy(self):
        """Assign a device id to the EngineArgs object."""
        from infinity_emb.inference import loading_strategy  # type: ignore

        object.__setattr__(self, "_loading_strategy", loading_strategy.get_loading_strategy(self))
        return self

    def copy(self):
        return deepcopy(self)

    @classmethod
    def from_env(cls) -> list["EngineArgs"]:
        """Create a list of EngineArgs from environment variables."""
        return [
            EngineArgs(
                model_name_or_path=model_name_or_path,
                batch_size=batch_size,
                revision=revision,
                trust_remote_code=trust_remote_code,
                engine=engine,
                model_warmup=model_warmup,
                device=device,
                compile=compile,
                bettertransformer=bettertransformer,
                dtype=dtype,
                pooling_method=pooling_method,
                lengths_via_tokenize=lengths_via_tokenize,
                embedding_dtype=embedding_dtype,
                served_model_name=served_model_name,
            )
            for model_name_or_path, batch_size, revision, trust_remote_code, engine, model_warmup, device, compile, bettertransformer, dtype, pooling_method, lengths_via_tokenize, embedding_dtype, served_model_name in zip_longest(
                MANAGER.model_id,
                MANAGER.batch_size,
                MANAGER.revision,
                MANAGER.trust_remote_code,
                MANAGER.engine,
                MANAGER.model_warmup,
                MANAGER.device,
                MANAGER.compile,
                MANAGER.bettertransformer,
                MANAGER.dtype,
                MANAGER.pooling_method,
                MANAGER.lengths_via_tokenize,
                MANAGER.embedding_dtype,
                MANAGER.served_model_name,
            )
        ]
