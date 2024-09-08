# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from infinity_emb._optional_imports import CHECK_PIL, CHECK_REQUESTS
from infinity_emb.primitives import ImageCorruption, ImageSingle
from typing import Union, List
import concurrent.futures

if CHECK_PIL.is_available:
    from PIL import Image  # type: ignore
if CHECK_REQUESTS.is_available:
    import requests  # type: ignore


def resolve_from_img_obj(img_obj):
    """Resolve an image from a PIL.Image.Image Object."""
    try:
        return ImageSingle(image=img_obj)
    except Exception as e:
        raise ImageCorruption(f"error opening image from obj: {e}")


def resolve_from_img_url(img_url):
    """Resolve an image from an URL."""
    try:
        downloaded_img = requests.get(img_url, stream=True).raw
    except Exception as e:
        raise ImageCorruption(f"error downloading image from url: {e}")
    
    try:
        return ImageSingle(image=Image.open(downloaded_img))
    except Exception as e:
        raise ImageCorruption(f"error opening image from url: {e}")


def resolve_image(img: Union[str, Image.Image]) -> ImageSingle:
    """Resolve a single image."""
    if isinstance(img, Image.Image):
        return resolve_from_img_obj(img)
    elif isinstance(img, str):
        return resolve_from_img_url(img)
    else:
        raise ValueError(f"Invalid image type: {img} is neither str nor PIL.Image.Image object")


def resolve_images(images: List[Union[str, Image.Image]], max_workers: int=10) -> List[ImageSingle]:
    """Resolve images from URLs or PIL.Image.Image Objects using multithreading."""
    CHECK_REQUESTS.mark_required()
    CHECK_PIL.mark_required()

    resolved_imgs = []
    exceptions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(resolve_image, img): img for img in images}
        for future in concurrent.futures.as_completed(futures):
            img = futures[future]
            try:
                resolved_imgs.append(future.result())
            except Exception as e:
                exceptions.append((img, str(e)))

    if exceptions:
        for img, error_msg in exceptions:
            print(f"Failed to resolve image: {img}.\nError msg: {error_msg}")
        raise ImageCorruption("One or more images failed to resolve. See details above.")

    return resolved_imgs