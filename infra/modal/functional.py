"""
Showcases the functional API of infinity via Modal.
Example is currently not tested.
"""

from modal import App, Image, build, enter, method
import os

INFINITY_VERSION = os.environ.get("INFINITY_VERSION", "0.0.63")
image = Image.from_registry(f"michaelf34/infinity:{INFINITY_VERSION}").entrypoint([])

app = App("infinity-functional", image=image)

with image.imports():
    from infinity_emb import AsyncEngineArray, EngineArgs


class BaseInfinityModel:
    """Base class for the infinity model."""

    def _get_array(self):
        return AsyncEngineArray.from_args(
            [
                EngineArgs(model_name_or_path=m, model_warmup=False)
                for m in self.model_id
            ]
        )

    @build()
    async def download_model(self):
        print(f"downloading models {self.model_id} ...")
        self._get_array()

    @enter()
    async def enter(self):
        print("Starting the engine array ...")
        self.engine_array = self._get_array()
        await self.engine_array.astart()
        print("engine array started!")


@app.cls(gpu="any", allow_concurrent_inputs=500)
class InfinityModal(BaseInfinityModel):
    """Model for embedding text (via clip or Bert) and images (via clip) ."""

    def __init__(self, model_id: tuple[str]) -> None:
        self.model_id = model_id
        super().__init__()

    @method()
    async def embed(self, sentences: list[str], model: str | int = 0):
        engine = self.engine_array[model]
        embeddings, usage = await engine.embed(sentences=sentences)
        return embeddings

    @method()
    async def image_embed(self, urls: list[str], model: str | int = 0):
        engine = self.engine_array[model]
        embeddings, usage = await engine.image_embed(images=urls)
        return embeddings

    @method()
    async def rerank(self, query: str, docs: list[str], model: str | int = 0):
        engine = self.engine_array[model]
        rankings, usage = await engine.rerank(query=query, docs=docs)
        return rankings

    @method()
    async def classify(self, sentences: list[str], model: str | int = 0):
        engine = self.engine_array[model]
        classes, usage = await engine.classify(sentences=sentences)
        return classes


@app.local_entrypoint()
def main():
    model_id = (
        "jinaai/jina-clip-v1",
        "michaelfeil/bge-small-en-v1.5",
        "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "philschmid/tiny-bert-sst2-distilled",
    )
    deployment = InfinityModal(model_id=model_id)
    embeddings_1 = deployment.embed.remote(sentences=["hello world"], model=model_id[1])
    embeddings_2 = deployment.image_embed.remote(
        urls=["http://images.cocodataset.org/val2017/000000039769.jpg"],
        model=model_id[0],
    )

    rerankings_1 = deployment.rerank.remote(
        query="Where is Paris?",
        docs=["Paris is the capital of France.", "Berlin is a city in Europe."],
        model=model_id[2],
    )

    classifications_1 = deployment.classify.remote(
        sentences=["I feel great today!"], model=model_id[3]
    )

    print(
        "Success, all tasks submitted! Embeddings:",
        embeddings_1[0].shape,
        embeddings_2[0].shape,
        "Rerankings:",
        rerankings_1,
        "Classifications:",
        classifications_1,
    )
