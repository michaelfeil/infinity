import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from infinity_emb.transformer.audio import utils as audio
from infinity_emb.transformer.vision import utils as vision


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "audio"])
@pytest.mark.parametrize("outcome", ["failure", "cancel", "success", "empty"])
async def test_media_tasks_finish_before_session_closes(monkeypatch, kind, outcome):
    module = vision if kind == "image" else audio
    session = SimpleNamespace(closed=False)
    started = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    children = []
    closed_during_cleanup = []

    @asynccontextmanager
    async def client_session():
        try:
            yield session
        finally:
            session.closed = True

    async def resolve(value, *args, **kwargs):
        children.append(asyncio.current_task())
        if value == "fail":
            await started.wait()
            raise ValueError("bad media")
        started.set()
        try:
            if outcome in ("failure", "cancel"):
                await release.wait()
            return value
        finally:
            await asyncio.sleep(0)
            closed_during_cleanup.append(session.closed)
            cleaned.set()

    monkeypatch.setattr(module, "media_client_session", client_session)
    monkeypatch.setattr(module, "resolve_image" if kind == "image" else "resolve_audio", resolve)
    values = [] if outcome == "empty" else ["slow", "fail" if outcome == "failure" else "last"]
    coro = (
        module.resolve_images(values) if kind == "image" else module.resolve_audios(values, 16000)
    )
    parent = asyncio.create_task(coro)
    try:
        if outcome == "cancel":
            await asyncio.wait_for(started.wait(), 1)
            parent.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(parent, 1)
        elif outcome == "failure":
            with pytest.raises(ValueError, match="bad media"):
                await asyncio.wait_for(parent, 1)
        else:
            assert await asyncio.wait_for(parent, 1) == values
        assert session.closed
        assert all(child.done() for child in children)
        assert not any(closed_during_cleanup)
        if values:
            assert cleaned.is_set()
    finally:
        release.set()
        if not parent.done():
            parent.cancel()
        await asyncio.gather(parent, *children, return_exceptions=True)
