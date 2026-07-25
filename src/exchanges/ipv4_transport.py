from __future__ import annotations

import socket

import aiohttp


class IPv4TransportMixin:
    """Keep an async CCXT client on a stable IPv4 source address."""

    def open(self):
        if not self.own_session or self.session is not None:
            return super().open()

        # Let CCXT initialize its event loop, throttler, and SSL context without
        # creating the default dual-stack connector.
        self.own_session = False
        try:
            super().open()
        finally:
            self.own_session = True

        self.tcp_connector = aiohttp.TCPConnector(
            ssl=self.ssl_context,
            loop=self.asyncio_loop,
            enable_cleanup_closed=True,
            family=socket.AF_INET,
        )
        self.session = aiohttp.ClientSession(
            loop=self.asyncio_loop,
            connector=self.tcp_connector,
            trust_env=self.aiohttp_trust_env,
        )
