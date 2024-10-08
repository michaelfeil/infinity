# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import asyncio
import io
from typing import Union

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


def resolve_from_img_bytes(bytes_img: bytes) -> ImageSingle:
    """Resolve an image from a Data URI"""
    try:
        img = Image.open(io.BytesIO(bytes_img))
        return ImageSingle(image=img)
    except Exception as e:
        raise ImageCorruption(f"error decoding data URI: {e}")


async def resolve_image(
    img: Union[str, "ImageClassType", bytes], session: "aiohttp.ClientSession"
) -> ImageSingle:
    """Resolve a single image."""
    if isinstance(img, Image.Image):
        return resolve_from_img_obj(img)
    elif isinstance(img, bytes):
        return resolve_from_img_bytes(img)
    elif isinstance(img, str):
        return await resolve_from_img_url(img, session=session)
    else:
        raise ValueError(
            f"Invalid image type: {img} is neither str nor ImageClassType object"
        )


async def resolve_images(
    images: list[Union[str, "ImageClassType", bytes]]
) -> list[ImageSingle]:
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
