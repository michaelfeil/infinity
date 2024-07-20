from embed._infer import BatchedInference
import importlib.metadata

__version__ = importlib.metadata.version("infinity_emb")

__all__ = ["BatchedInference", "__version__"]
