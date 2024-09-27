# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import asyncio
import re
import io
from base64 import b64decode
from typing import List, Union

from infinity_emb._optional_imports import CHECK_AIOHTTP, CHECK_PIL
from infinity_emb.primitives import (
    ImageClassType,
    ImageCorruption,
    ImageSingle,
)

if CHECK_AIOHTTP.is_available:
    import aiohttp

if CHECK_PIL.is_available:
    from PIL import Image  # type: ignore


def resolve_from_img_obj(img_obj: "ImageClassType") -> ImageSingle:
    """Resolve an image from a ImageClassType Object."""
    return ImageSingle(image=img_obj)


async def resolve_from_img_url(
    img_url: str, session: "aiohttp.ClientSession"
) -> ImageSingle:
    """Resolve an image from an URL."""
    try:
        # requests.get(img_url, stream=True).raw
        downloaded_img = await (await session.get(img_url)).read()
    except Exception as e:
        raise ImageCorruption(
            f"error opening an image in your request image from url: {e}"
        )

    try:
        img = Image.open(io.BytesIO(downloaded_img))
        if img.size[0] < 3 or img.size[1] < 3:
            # https://upload.wikimedia.org/wikipedia/commons/c/ca/1x1.png
            raise ImageCorruption(
                f"An image in your request is too small for processing {img.size}"
            )
        return ImageSingle(image=img)
    except Exception as e:
        raise ImageCorruption(
            f"error opening the payload from an image in your request from url: {e}"
        )

def resolve_from_img_base64(uri: str) -> ImageSingle:
    """Resolve an image from a Data URI"""
    try:
        base64_image = uri.split(",")[-1]
        decoded_image = b64decode(base64_image)
        img = Image.open(io.BytesIO(decoded_image))
        return ImageSingle(image=img)
    except Exception as e:
        raise ImageCorruption(
            f"error decoding data URI: {e}"
        )


def is_base64_check(s: str):
    """Regex check to quickly check if string is base64 or not."""
    pattern = (
        r"^[A-Za-z0-9+/]{4}([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$"
    )
    return bool(re.match(pattern, s))


def is_base64_data_uri(uri: str) -> bool:
    """Simply check if the uri is a Data URI or not

    Ref: https://developer.mozilla.org/en-US/docs/web/http/basics_of_http/data_urls
    """

    starts_with_data = uri.startswith("data:")

    b64_data_q = uri.split(",")[-1]
    is_base64 = is_base64_check(b64_data_q)

    return starts_with_data and is_base64



async def resolve_image(
    img: Union[str, "ImageClassType"], session: "aiohttp.ClientSession"
) -> ImageSingle:
    """Resolve a single image."""
    if isinstance(img, Image.Image):
        return resolve_from_img_obj(img)
    elif is_base64_data_uri(img):
        return resolve_from_img_base64(img)
    elif isinstance(img, str):
        return await resolve_from_img_url(img, session=session)
    else:
        raise ValueError(
            f"Invalid image type: {img} is neither str nor ImageClassType object"
        )


async def resolve_images(
    images: List[Union[str, "ImageClassType"]]
) -> List[ImageSingle]:
    """Resolve images from URLs or ImageClassType Objects using multithreading."""
    # TODO: improve parallel requests, safety, error handling
    CHECK_AIOHTTP.mark_required()
    CHECK_PIL.mark_required()

    resolved_imgs = []

    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            resolved_imgs = await asyncio.gather(
                *[resolve_image(img, session) for img in images]
            )
    except Exception as e:
        raise ImageCorruption(
            f"Failed to resolve image: {images}.\nError msg: {str(e)}"
        )

    return resolved_imgs
