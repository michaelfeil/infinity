import asyncio
import io
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image, ImageFile
from transformers import CLIPImageProcessor

from infinity_emb.inference import BatchHandler, caching_layer
from infinity_emb.primitives import ImageCorruption


@pytest.fixture
def png_bytes():
    pixels = np.random.default_rng(0).integers(256, size=(512, 512, 3), dtype=np.uint8)
    with io.BytesIO() as buffer:
        Image.fromarray(pixels).save(buffer, format="PNG")
        return buffer.getvalue()


@pytest.fixture
def image_handler(tmp_path, monkeypatch):
    monkeypatch.setattr(caching_layer.MANAGER, "cache_dir", tmp_path)
    handlers = []

    def create(model=None, cached=True):
        handler = BatchHandler(
            model_replicas=[model or SimpleNamespace(capabilities={"image_embed"})],
            max_batch_size=2,
            vector_disk_cache_path="images" if cached else "",
        )
        handlers.append(handler)
        return handler

    yield create
    for handler in handlers:
        handler._shutdown.set()
        handler._threadpool.shutdown(wait=True)
        cache = handler._result_store._cache
        if cache is not None:
            cache._threadpool.shutdown(wait=True)
            cache._cache.close()


@pytest.mark.anyio
@pytest.mark.parametrize("input_kind", ["image", "bytes"])
async def test_lazy_image_cache_and_preprocessing_overlap(
    image_handler, png_bytes, monkeypatch, input_kind
):
    barrier = threading.Barrier(2)
    loaded_at_dispatch = []
    decode_threads = []
    loop_thread = threading.get_ident()
    original_load = ImageFile.ImageFile.load

    def tracked_load(image, *args, **kwargs):
        if image.tile:
            decode_threads.append(threading.get_ident())
        return original_load(image, *args, **kwargs)

    monkeypatch.setattr(ImageFile.ImageFile, "load", tracked_load)

    class ImageModel:
        capabilities = {"image_embed"}
        processor = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size=32)

        def encode_pre(self, images):
            loaded_at_dispatch.append(not images[0].tile)
            barrier.wait(timeout=5)
            return self.processor(images=images, return_tensors="np")["pixel_values"]

        def encode_core(self, features):
            # Deterministic inference fixture; preprocessing is the real CLIP processor.
            return features.mean(axis=(2, 3))

        def encode_post(self, features):
            return list(features)

    handler = image_handler(model=ImageModel())
    cache = handler._result_store._cache
    original_get = cache._get
    cache_hits = []
    cache_tasks = []

    def overlapping_get(content):
        loaded_at_dispatch.append(not content.image.tile)
        barrier.wait(timeout=5)
        key, result = original_get(content)
        cache_hits.append(result is not None)
        return key, result

    original_complete = cache.aget_complete

    def wait_for_cache_writes():
        with cache._add_q.all_tasks_done:
            assert cache._add_q.all_tasks_done.wait_for(
                lambda: cache._add_q.unfinished_tasks == 0, timeout=5
            )

    async def tracked_complete(item):
        cache_tasks.append(asyncio.current_task())
        await original_complete(item)

    monkeypatch.setattr(cache, "_get", overlapping_get)
    monkeypatch.setattr(cache, "aget_complete", tracked_complete)
    await handler.spawn()
    try:
        results = []
        for _ in range(2):
            with Image.open(io.BytesIO(png_bytes)) as image:
                assert image.tile  # The fixture starts with undecoded pixels.
                source = image if input_kind == "image" else png_bytes
                result, _ = await asyncio.wait_for(handler.image_embed(images=[source]), timeout=5)
                await asyncio.wait_for(asyncio.gather(*cache_tasks), timeout=5)
                await asyncio.to_thread(wait_for_cache_writes)
                results.append(result[0])
        np.testing.assert_array_equal(results[0], results[1])
        assert cache_hits == [False, True]
        # A barrier makes both readers observe preparation before either reads pixels.
        assert loaded_at_dispatch == [True] * 4
        assert len(decode_threads) == 2
        assert all(thread != loop_thread for thread in decode_threads)
    finally:
        await handler.shutdown()
        await asyncio.wait_for(asyncio.gather(*cache_tasks, return_exceptions=True), timeout=5)


@pytest.mark.anyio
async def test_uncached_images_remain_lazy_until_preprocessing(
    image_handler, png_bytes, monkeypatch
):
    handler = image_handler(cached=False)
    schedule = AsyncMock(return_value=([], 0))
    monkeypatch.setattr(handler, "_schedule", schedule)
    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.tile
        assert await handler.image_embed(images=[image]) == ([], 0)
        assert image.tile
        assert schedule.call_args.args[0][0].image is image


@pytest.mark.anyio
async def test_cached_image_decode_failure_is_reported_before_dispatch(
    image_handler, png_bytes, monkeypatch
):
    handler = image_handler()
    schedule = AsyncMock(return_value=([], 0))
    monkeypatch.setattr(handler, "_schedule", schedule)
    with pytest.raises(ImageCorruption, match="error decoding image pixels") as error:
        await handler.image_embed(images=[png_bytes[: len(png_bytes) // 2]])
    assert isinstance(error.value.__cause__, OSError)
    schedule.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_first", [False, True], ids=["concurrent", "cancelled"])
async def test_cached_image_preparation_serializes_shared_images(
    image_handler, png_bytes, monkeypatch, cancel_first
):
    handler = image_handler()
    schedule = AsyncMock(return_value=([], 0))
    monkeypatch.setattr(handler, "_schedule", schedule)
    decode_started = threading.Event()
    decode_release = threading.Event()
    lock_contended = threading.Event()

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                lock_contended.set()
                self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(handler, "_image_load_lock", ObservedLock())
    with Image.open(io.BytesIO(png_bytes)) as image:
        original_load = image.load
        decode_count = 0

        def paused_load():
            nonlocal decode_count
            if image.tile:
                decode_count += 1
                decode_started.set()
                assert decode_release.wait(timeout=5)
            return original_load()

        monkeypatch.setattr(image, "load", paused_load)
        first = asyncio.create_task(handler.image_embed(images=[image, image]))
        second = None
        try:
            assert await asyncio.to_thread(decode_started.wait, 3)
            if cancel_first:
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            second = asyncio.create_task(handler.image_embed(images=[image]))
            # The first decode still owns the lock after cancellation of its awaiter.
            assert await asyncio.to_thread(lock_contended.wait, 3)
            assert decode_count == 1
            decode_release.set()
            await asyncio.wait_for(second, timeout=5)
            if not cancel_first:
                await asyncio.wait_for(first, timeout=5)
            assert decode_count == 1
            assert not image.tile
            assert schedule.await_count == (1 if cancel_first else 2)
        finally:
            decode_release.set()
            await asyncio.wait_for(
                asyncio.gather(first, *([second] if second else []), return_exceptions=True),
                timeout=5,
            )
