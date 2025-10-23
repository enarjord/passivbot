"""Tests for realtime snapshot fetching behaviour."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List

from risk_management.configuration import AccountConfig, RealtimeConfig
from risk_management.realtime import AuthenticationError, RealtimeDataFetcher


class StubAccountClient:
    """Test double for ``AccountClientProtocol`` returning predefined outcomes."""

    def __init__(self, outcomes: Iterable[object]):
        self._outcomes: List[object] = list(outcomes)

    async def fetch(self):  # pragma: no cover - exercised through tests
        if not self._outcomes:
            raise RuntimeError("No more outcomes configured for StubAccountClient")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def close(self) -> None:  # pragma: no cover - included for interface completeness
        return None


def _make_config() -> RealtimeConfig:
    return RealtimeConfig(accounts=[AccountConfig(name="Test", exchange="test")])


def test_authentication_warning_logged_once(caplog) -> None:
    client = StubAccountClient(
        [AuthenticationError("invalid"), AuthenticationError("invalid")]
    )
    fetcher = RealtimeDataFetcher(_make_config(), account_clients=[client])

    caplog.set_level(logging.WARNING)

    snapshot_first = asyncio.run(fetcher.fetch_snapshot())
    snapshot_second = asyncio.run(fetcher.fetch_snapshot())

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert snapshot_first["account_messages"]["Test"].startswith(
        "Test: authentication failed"
    )
    assert snapshot_second["account_messages"]["Test"].startswith(
        "Test: authentication failed"
    )
    asyncio.run(fetcher.close())


def test_authentication_warning_resets_after_success(caplog) -> None:
    client = StubAccountClient(
        [
            AuthenticationError("invalid"),
            {"name": "Test", "balance": 123.0, "positions": []},
            AuthenticationError("invalid"),
        ]
    )
    fetcher = RealtimeDataFetcher(_make_config(), account_clients=[client])

    caplog.set_level(logging.INFO)

    asyncio.run(fetcher.fetch_snapshot())
    snapshot_success = asyncio.run(fetcher.fetch_snapshot())
    asyncio.run(fetcher.fetch_snapshot())

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    infos = [record for record in caplog.records if record.levelno == logging.INFO]

    assert len(warnings) == 2
    assert any("Authentication for Test restored" in record.message for record in infos)
    assert snapshot_success["accounts"][0]["balance"] == 123.0
    asyncio.run(fetcher.close())
