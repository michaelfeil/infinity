import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from infinity_emb.transformer._media_url import (
    download_media_url,
    media_client_session,
    media_url_blacklist,
)
from yarl import URL


def test_default_blacklist(monkeypatch):
    monkeypatch.delenv("INFINITY_MEDIA_URL_BLACKLIST", raising=False)
    policy = media_url_blacklist()
    for host in [
        "localhost",
        "localhost.",
        "test.localhost",
        "127.0.0.1",
        "10.1.2.3",
        "172.16.1.1",
        "192.168.1.1",
        "169.254.1.1",
        "[::1]",
        "[fc00::1]",
        "[fe80::1]",
        "[::ffff:127.0.0.1]",
    ]:
        with pytest.raises(ValueError, match="blacklisted"):
            policy.validate_url(f"http://{host}/image.png")
    for url in ["https://images.example/photo.png", "http://8.8.8.8/image.png"]:
        assert str(policy.validate_url(url)) == url
    for url in [
        "file:///tmp/image.png",
        "ftp://images.example/photo.png",
        "https://user:password@images.example/photo.png",
    ]:
        with pytest.raises(ValueError):
            policy.validate_url(url)


def test_configurable_blacklist(monkeypatch):
    monkeypatch.setenv(
        "INFINITY_MEDIA_URL_BLACKLIST",
        "*.blocked.example;203.0.113.0/24;https://images.example/private/",
    )
    policy = media_url_blacklist()
    for url in [
        "https://a.blocked.example/a",
        "http://203.0.113.1/a",
        "https://images.example/private/photo.png",
    ]:
        with pytest.raises(ValueError, match="blacklisted"):
            policy.validate_url(url)
    assert policy.validate_url("https://images.example/public/photo.png")
    assert policy.validate_url("http://10.0.0.1/photo.png")  # Explicit replacement of defaults.
    monkeypatch.setenv("INFINITY_MEDIA_URL_BLACKLIST", "")
    assert media_url_blacklist().validate_url("http://localhost/photo.png")


def test_connection_dns_is_checked(monkeypatch):
    monkeypatch.delenv("INFINITY_MEDIA_URL_BLACKLIST", raising=False)
    resolve = AsyncMock(return_value=[{"host": "8.8.8.8"}, {"host": "127.0.0.1"}])
    monkeypatch.setattr("aiohttp.resolver.ThreadedResolver.resolve", resolve)

    async def run():
        async with media_client_session() as session:
            assert not session.trust_env
            with pytest.raises(ValueError, match="blacklisted"):
                await session.connector._resolver.resolve("images.example", 443)
            resolve.return_value = [{"host": "8.8.8.8"}]
            assert await session.connector._resolver.resolve("images.example", 443) == [
                {"host": "8.8.8.8"}
            ]

    asyncio.run(run())


def test_redirects_are_revalidated(monkeypatch):
    monkeypatch.delenv("INFINITY_MEDIA_URL_BLACKLIST", raising=False)

    def response(status, location=None):
        result = MagicMock(status=status, url=URL("https://images.example/original"))
        result.headers = {"Location": location} if location else {}
        result.read = AsyncMock(return_value=b"image payload")
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=result)
        context.__aexit__ = AsyncMock(return_value=False)
        return context

    async def run():
        session = MagicMock()
        session.get.return_value = response(302, "http://127.0.0.1/private")
        with pytest.raises(ValueError, match="blacklisted"):
            await download_media_url("https://images.example/original", session)
        assert session.get.call_count == 1
        session.get.reset_mock()
        session.get.side_effect = [response(302, "/photo.png"), response(200)]
        assert (
            await download_media_url("https://images.example/original", session) == b"image payload"
        )
        assert str(session.get.call_args.args[0]) == "https://images.example/photo.png"
        assert session.get.call_args.kwargs == {"allow_redirects": False}

    asyncio.run(run())
