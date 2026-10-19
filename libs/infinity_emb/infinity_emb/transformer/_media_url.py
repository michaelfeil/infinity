# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""Blacklist checks shared by client-provided image and audio downloads."""

from __future__ import annotations

import ipaddress
import os
import socket
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp
    from yarl import URL

DEFAULT_MEDIA_URL_BLACKLIST = (
    "localhost;*.localhost;0.0.0.0/8;10.0.0.0/8;100.64.0.0/10;127.0.0.0/8;"
    "169.254.0.0/16;172.16.0.0/12;192.168.0.0/16;224.0.0.0/4;"
    "::/128;::1/128;fc00::/7;fe80::/10;ff00::/8"
)


class MediaURLBlacklist:
    def __init__(self, rules: str):
        self.networks = []
        self.hosts = []
        for rule in filter(None, (value.strip() for value in rules.split(";"))):
            try:
                self.networks.append(ipaddress.ip_network(rule, strict=False))
            except ValueError:
                if "/" in rule or ":" in rule:
                    raise ValueError("Invalid INFINITY_MEDIA_URL_BLACKLIST entry") from None
                self.hosts.append(rule.lower().rstrip("."))

    def validate_host(self, host: str) -> None:
        host = host.lower().rstrip(".")
        if any(fnmatchcase(host, pattern) for pattern in self.hosts):
            raise ValueError("Media URL host is blacklisted")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            from aiohttp.helpers import is_ip_address

            if not is_ip_address(host):
                return  # Hostname addresses are checked by the connection's resolver.
            # aiohttp also skips DNS for legacy numeric IPv4 forms. Normalize
            # those before matching CIDRs; inet_aton parses without DNS/I/O.
            try:
                address = ipaddress.IPv4Address(socket.inet_aton(host))
            except OSError:
                raise ValueError("Invalid media URL IP address") from None
        addresses = [address]
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            addresses.append(address.ipv4_mapped)
        if any(
            candidate.version == network.version and candidate in network
            for candidate in addresses
            for network in self.networks
        ):
            raise ValueError("Media URL address is blacklisted")

    def validate_url(self, value: str) -> URL:
        from yarl import URL

        url = URL(value)
        if url.scheme not in {"http", "https"} or not url.host or url.user is not None:
            raise ValueError("Media URLs must be HTTP(S) URLs without credentials")
        if "%" in url.host:
            raise ValueError("Scoped or escaped media URL hosts are not supported")
        self.validate_host(url.host)
        return url


def media_url_blacklist() -> MediaURLBlacklist:
    return MediaURLBlacklist(
        os.environ.get("INFINITY_MEDIA_URL_BLACKLIST", DEFAULT_MEDIA_URL_BLACKLIST)
    )


def media_client_session() -> aiohttp.ClientSession:
    import aiohttp
    from aiohttp.resolver import ThreadedResolver

    blacklist = media_url_blacklist()

    class MediaResolver(ThreadedResolver):
        async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
            blacklist.validate_host(host)
            records = await super().resolve(host, port, family)
            # Validate the exact records used to connect, not a separate DNS lookup.
            for record in records:
                blacklist.validate_host(record["host"])
            return records

    # Proxy-side DNS would bypass the resolver checks. Do not use ambient proxies/netrc.
    connector = aiohttp.TCPConnector(resolver=MediaResolver())
    return aiohttp.ClientSession(connector=connector, trust_env=False)


async def download_media_url(value: str, session: aiohttp.ClientSession) -> bytes:
    from yarl import URL

    blacklist = media_url_blacklist()
    url = blacklist.validate_url(value)
    for _ in range(11):
        async with session.get(url, allow_redirects=False) as response:
            if response.status in {301, 302, 303, 307, 308} and "Location" in response.headers:
                url = blacklist.validate_url(
                    str(response.url.join(URL(response.headers["Location"])))
                )
                continue
            response.raise_for_status()
            return await response.read()
    raise ValueError("Too many media URL redirects")
