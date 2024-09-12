# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from infinity_emb._optional_imports import CHECK_PIL, CHECK_REQUESTS
from infinity_emb.primitives import ImageCorruption, ImageSingle
from typing import Union, List, Sequence

if CHECK_PIL.is_available:
    from PIL import Image  # type: ignore
if CHECK_REQUESTS.is_available:
    import requests  # type: ignore


def resolve_from_img_obj(img_obj):
    """Resolve an image from a PIL.Image.Image Object."""
    return ImageSingle(image=img_obj)


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
        raise ValueError(
            f"Invalid image type: {img} is neither str nor PIL.Image.Image object"
        )


def resolve_images(images: Sequence[Union[str, Image.Image]]) -> List[ImageSingle]:
    """Resolve images from URLs or PIL.Image.Image Objects using multithreading."""
    # TODO: improve parallel requests, safety, error handling
    CHECK_REQUESTS.mark_required()
    CHECK_PIL.mark_required()

    resolved_imgs = []
    for img in images:
        try:
            resolved_imgs.append(resolve_image(img))
        except Exception as e:
            raise ImageCorruption(
                f"Failed to resolve image: {img}.\nError msg: {str(e)}"
            )

    return resolved_imgs
