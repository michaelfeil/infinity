"""
Showcases the functional API of infinity via Modal.
Example is currently not tested.
"""

from modal import App, Image, build, enter, method

image = Image.from_registry("michaelf34/infinity:0.0.45").entrypoint([])

app = App("infinity-functional", image=image)

with image.imports():
    from infinity_emb import AsyncEngineArray, EngineArgs


class BaseInfinityModel:
    """Base class for the infinity model."""
    model_id: tuple[str]

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


@app.cls(gpu="any")
class InfinityModal(BaseInfinityModel):
    """Model for embedding text (via clip or Bert) and images (via clip) ."""

    model_id = (
        "jinaai/jina-clip-v1",
        "michaelfeil/bge-small-en-v1.5",
    )

    @method()
    async def embed(self, sentences: list[str], model: str | int = 0):
        engine = self.engine_array[model]
        embeddings, usage = await engine.embed(sentences)
        return embeddings

    @method()
    async def image_embed(self, urls: list[str], model: str | int = 0):
        engine = self.engine_array[model]
        embeddings, usage = await engine.image_embed(urls)
        return embeddings


@app.cls(gpu="any")
class InfinityRerankModal(BaseInfinityModel):
    """Model for re-ranking documents."""

    model_id = ("mixedbread-ai/mxbai-rerank-large-v1",)

    @method()
    async def rerank(self, query: str, docs: list[str], model: str | int = 2):
        engine = self.engine_array[model]
        rankings, usage = await engine.rerank(query=query, docs=docs)
        return rankings


@app.cls(gpu="any")
class InfinityClassifyModal(BaseInfinityModel):
    """Model for classifying text into 51 languages."""

    model_id = ("qanastek/51-languages-classifier",)

    @method()
    async def classify(self, texts: list[str], model: str | int = 3):
        engine = self.engine_array[model]
        classes, usage = await engine.classify(texts)
        return classes
