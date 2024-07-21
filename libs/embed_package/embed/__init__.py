import importlib.metadata

from embed._infer import BatchedInference

__version__ = importlib.metadata.version("infinity_emb")

__all__ = ["BatchedInference", "__version__"]
