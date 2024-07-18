from infinity_emb._optional_imports import CHECK_PIL, CHECK_REQUESTS
from infinity_emb.primitives import ImageCorruption, ImageSingle

if CHECK_PIL.is_available:
    from PIL import Image  # type: ignore
if CHECK_REQUESTS.is_available:
    import requests  # type: ignore


def resolve_images(image_urls: list[str]) -> list[ImageSingle]:
    """Resolve images from URLs."""
    # TODO: improve parallel requests, safety, error handling
    CHECK_REQUESTS.mark_required()
    CHECK_PIL.mark_required()

    try:
        downloaded = [requests.get(url, stream=True).raw for url in image_urls]
    except Exception as e:
        raise ImageCorruption(f"Error downloading images: {e}")
    try:
        return [ImageSingle(image=Image.open(content)) for content in downloaded]
    except Exception as e:
        raise ImageCorruption(f"Error opening images: {e}")
