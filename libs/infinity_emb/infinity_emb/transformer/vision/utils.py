import requests
from PIL import Image

from infinity_emb.primitives import ImageSingle


def resolve_images(image_urls) -> list[ImageSingle]:
    # TODO: improve parallel requests, safety, error handling
    return [
        ImageSingle(image=Image.open(requests.get(url, stream=True).raw))
        for url in image_urls
    ]
