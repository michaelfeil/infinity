import asyncio
import threading

import numpy as np
import pytest
from PIL import Image

from infinity_emb.inference import caching_layer
from infinity_emb.primitives import (
    AudioSingle,
    EmbeddingInner,
    EmbeddingSingle,
    ImageSingle,
    PredictSingle,
    ReRankSingle,
    get_inner_item,
)


@pytest.mark.anyio
async def test_cache():
    global INFINITY_CACHE_VECTORS

    loop = asyncio.get_event_loop()
    shutdown = threading.Event()
    try:
        INFINITY_CACHE_VECTORS = True
        sentence = "dummy"
        embedding = np.random.random(5).tolist()
        c = caching_layer.Cache(
            cache_name=f"pytest_{hash((sentence, tuple(embedding)))}", shutdown=shutdown
        )

        sample = EmbeddingInner(
            content=EmbeddingSingle(sentence=sentence), future=loop.create_future()
        )
        sample_embedded = EmbeddingInner(
            content=EmbeddingSingle(sentence=sentence),
            future=loop.create_future(),
            embedding=None,
        )
        await sample_embedded.complete(embedding)
        await c.aget_complete(sample_embedded)
        # add the embedded sample
        await asyncio.sleep(0.5)
        # launch the ba
        await c.aget_complete(sample)
        assert sample.future.done()
        assert sample.embedding is not None
        np.testing.assert_array_equal(sample.embedding, embedding)
    finally:
        INFINITY_CACHE_VECTORS = False
        shutdown.set()


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(caching_layer.MANAGER, "cache_dir", tmp_path)
    shutdown = threading.Event()
    cache = caching_layer.Cache(cache_name="content_keys", shutdown=shutdown)
    try:
        yield cache
    finally:
        shutdown.set()
        cache._threadpool.shutdown(wait=True)
        cache._cache.close()


async def lookup_or_infer(cache, content, value, monkeypatch):
    """Run real cache reads/writes, completing inference only on a cache miss."""
    item = get_inner_item(type(content))(
        content=content, future=asyncio.get_running_loop().create_future()
    )
    get_result = item.get_result

    async def infer():
        if isinstance(content, PredictSingle):
            result = {"label": "test", "score": value}
        elif isinstance(content, ReRankSingle):
            result = value
        else:
            result = np.full(3, value, dtype=np.float32)
        await item.complete(result)
        return await get_result()

    monkeypatch.setattr(item, "get_result", infer)
    await cache.aget_complete(item)
    await asyncio.to_thread(cache._add_q.join)
    result = item.future.result()
    if isinstance(content, PredictSingle):
        return result["score"]
    if isinstance(content, ReRankSingle):
        return result
    return result[0]


def palette_image(color):
    image = Image.new("P", (4, 4))
    image.putpalette(list(color) + [0] * (768 - 3))
    return image


def alpha_palette_image(alpha):
    image = Image.new("P", (4, 4))
    image.putpalette([255, 0, 0, alpha] * 256, rawmode="RGBA")
    return image


def transparent_image(transparent):
    image = palette_image((255, 0, 0))
    image.info["transparency"] = transparent
    return image


@pytest.mark.anyio
@pytest.mark.parametrize(
    "first, second",
    [
        (ReRankSingle(query="ab", document="c"), ReRankSingle(query="a", document="bc")),
        (
            ImageSingle(image=Image.new("RGB", (4, 4), "red")),
            ImageSingle(image=Image.new("RGB", (4, 4), "blue")),
        ),
        (
            ImageSingle(image=Image.new("RGB", (4, 4))),
            ImageSingle(image=Image.new("RGB", (8, 4))),
        ),
        (
            ImageSingle(image=Image.new("RGB", (4, 4))),
            ImageSingle(image=Image.new("HSV", (4, 4))),
        ),
        (
            ImageSingle(image=palette_image((255, 0, 0))),
            ImageSingle(image=palette_image((0, 0, 255))),
        ),
        (
            ImageSingle(image=alpha_palette_image(0)),
            ImageSingle(image=alpha_palette_image(255)),
        ),
        (
            ImageSingle(image=transparent_image(0)),
            ImageSingle(image=transparent_image(1)),
        ),
        (
            AudioSingle(audio=np.zeros(8), sampling_rate=16000),
            AudioSingle(audio=np.ones(8), sampling_rate=16000),
        ),
        (
            AudioSingle(audio=np.zeros(8), sampling_rate=16000),
            AudioSingle(audio=np.zeros(8), sampling_rate=48000),
        ),
        (
            AudioSingle(audio=np.zeros(8), sampling_rate=16000),
            AudioSingle(audio=np.zeros((8, 2)), sampling_rate=16000),
        ),
        (
            AudioSingle(audio=np.zeros(8, dtype=np.float32), sampling_rate=16000),
            AudioSingle(audio=np.zeros(8, dtype=np.float64), sampling_rate=16000),
        ),
        (
            ImageSingle(image=Image.new("RGB", (4, 4))),
            EmbeddingSingle(sentence="an image is worth a repeated " + "token" * 4),
        ),
        (EmbeddingSingle(sentence="hello"), PredictSingle(sentence="hello")),
    ],
    ids=[
        "rerank-boundary",
        "image-pixels",
        "image-size",
        "image-mode",
        "image-palette",
        "image-palette-alpha",
        "image-transparency",
        "audio-samples",
        "audio-rate",
        "audio-shape",
        "audio-dtype",
        "image-text-modality",
        "embedding-classification-modality",
    ],
)
async def test_cache_does_not_reuse_distinct_inputs(isolated_cache, monkeypatch, first, second):
    # These inputs all shared the old scheduling-string cache key.
    assert first.str_repr() == second.str_repr()
    assert await lookup_or_infer(isolated_cache, first, 1.0, monkeypatch) == 1.0
    assert await lookup_or_infer(isolated_cache, second, 2.0, monkeypatch) == 2.0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "make_content",
    [
        lambda: EmbeddingSingle(sentence="hello"),
        lambda: PredictSingle(sentence="hello"),
        lambda: ReRankSingle(query="hello", document="world"),
        lambda: ImageSingle(image=Image.new("RGB", (4, 4), "red")),
        lambda: ImageSingle(image=palette_image((255, 0, 0))),
        lambda: ImageSingle(image=alpha_palette_image(0)),
        lambda: AudioSingle(audio=np.arange(8, dtype=np.float32), sampling_rate=16000),
    ],
    ids=["text", "classification", "rerank", "image", "palette-image", "alpha-palette", "audio"],
)
async def test_cache_reuses_equivalent_inputs(isolated_cache, monkeypatch, make_content):
    assert await lookup_or_infer(isolated_cache, make_content(), 1.0, monkeypatch) == 1.0
    # A newly constructed equivalent input must hit the cached value, not infer 2.0.
    assert await lookup_or_infer(isolated_cache, make_content(), 2.0, monkeypatch) == 1.0


@pytest.mark.anyio
async def test_legacy_cache_entries_are_not_reused_or_removed(isolated_cache, monkeypatch):
    content = EmbeddingSingle(sentence="hello")
    isolated_cache._cache.add(content.str_repr(), np.full(3, 1.0, dtype=np.float32))
    assert await lookup_or_infer(isolated_cache, content, 2.0, monkeypatch) == 2.0
    np.testing.assert_array_equal(isolated_cache._cache.get(content.str_repr()), np.full(3, 1.0))


@pytest.mark.anyio
async def test_cache_key_is_computed_once_per_lookup(isolated_cache, monkeypatch):
    original_hash = isolated_cache._pre_hash
    calls = []

    def tracked_hash(content):
        calls.append(content)
        return original_hash(content)

    monkeypatch.setattr(isolated_cache, "_pre_hash", tracked_hash)
    content = EmbeddingSingle(sentence="hello")
    assert await lookup_or_infer(isolated_cache, content, 1.0, monkeypatch) == 1.0
    assert calls == [content]


@pytest.mark.anyio
@pytest.mark.parametrize("rerank", [False, True], ids=["text", "rerank"])
async def test_equal_strings_ignore_object_identity(isolated_cache, monkeypatch, rerank):
    text = "sentence"
    equal_text = "".join(["sen", "tence"])
    assert text == equal_text and text is not equal_text
    if rerank:
        first = ReRankSingle(query=text, document=text)
        second = ReRankSingle(query=text, document=equal_text)
    else:
        first = EmbeddingSingle(sentence=text)
        second = EmbeddingSingle(sentence=equal_text)
    assert await lookup_or_infer(isolated_cache, first, 1.0, monkeypatch) == 1.0
    assert await lookup_or_infer(isolated_cache, second, 2.0, monkeypatch) == 1.0


@pytest.mark.anyio
async def test_equal_audio_ignores_memory_layout(isolated_cache, monkeypatch):
    data = np.arange(8, dtype=np.float32).reshape(4, 2)
    first = AudioSingle(audio=np.array(data, order="C"), sampling_rate=16000)
    second = AudioSingle(audio=np.array(data, order="F"), sampling_rate=16000)
    assert await lookup_or_infer(isolated_cache, first, 1.0, monkeypatch) == 1.0
    assert await lookup_or_infer(isolated_cache, second, 2.0, monkeypatch) == 1.0


@pytest.mark.anyio
async def test_equal_image_ignores_metadata_order(isolated_cache, monkeypatch):
    first = ImageSingle(image=Image.new("RGB", (4, 4)))
    second = ImageSingle(image=Image.new("RGB", (4, 4)))
    first.image.info.update({"comment": "example", "dpi": (96, 96)})
    second.image.info.update({"dpi": (96, 96), "comment": "example"})
    assert await lookup_or_infer(isolated_cache, first, 1.0, monkeypatch) == 1.0
    assert await lookup_or_infer(isolated_cache, second, 2.0, monkeypatch) == 1.0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "make_content",
    [
        lambda text: EmbeddingSingle(sentence=text),
        lambda text: PredictSingle(sentence=text),
        lambda text: ReRankSingle(query=text, document="fixed"),
        lambda text: ReRankSingle(query="fixed", document=text),
    ],
    ids=["embedding", "classification", "rerank-query", "rerank-document"],
)
@pytest.mark.parametrize(
    "first, second",
    [("\U0001f600", "\ud83d\ude00"), ("\ud83d", "\\ud83d"), ("\ud83d", "\ude00")],
    ids=["non-bmp-surrogate-pair", "surrogate-escape", "lone-surrogates"],
)
async def test_cache_preserves_unicode_codepoints(
    isolated_cache, monkeypatch, make_content, first, second
):
    assert first != second
    assert await lookup_or_infer(isolated_cache, make_content(first), 1.0, monkeypatch) == 1.0
    assert await lookup_or_infer(isolated_cache, make_content(second), 2.0, monkeypatch) == 2.0
    assert await lookup_or_infer(isolated_cache, make_content(first), 3.0, monkeypatch) == 1.0
    assert await lookup_or_infer(isolated_cache, make_content(second), 4.0, monkeypatch) == 2.0
