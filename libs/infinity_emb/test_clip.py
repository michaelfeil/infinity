import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine

from PIL import Image
import requests

sentences = ["This is awesome.", "I am bored."]
# images = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img_obj = requests.get(url=img_url, stream=True).raw

img = Image.open(img_obj)

# images = [Image.open("http://images.cocodataset.org/val2017/000000039769.jpg")]
engine_args = EngineArgs(
    model_name_or_path = "/Users/wanggao/Public/infinity/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", 
    engine="torch"
)
array = AsyncEngineArray.from_args([engine_args])

images = ["http://images.cocodataset.org/val2017/000000039769.jpg",
          "http://images.cocodataset.org/val2017/000000039769.jpg",
          img]

async def embed(engine: AsyncEmbeddingEngine): 
    await engine.astart()
    embeddings, usage = await engine.embed(sentences=sentences)
    embeddings_image, _ = await engine.image_embed(images=images)
    # print("embeddings", [item.shape for item in embeddings])
    # # print("usage", usage)
    # print("embeddings_image", [item.shape for item in embeddings_image])
    # print(embeddings_image[0] == embeddings_image[2])
    await engine.astop()

asyncio.run(embed(array["/Users/wanggao/Public/infinity/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"]))