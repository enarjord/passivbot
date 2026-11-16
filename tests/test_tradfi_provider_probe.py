from unittest.mock import AsyncMock, Mock

import pytest

from tools import probe_tradfi_providers as probe


@pytest.mark.asyncio
async def test_probe_uses_selected_provider_for_every_requested_check(monkeypatch):
    provider = object()
    factory = Mock(return_value=provider)
    basic = AsyncMock(return_value=True)
    depth = AsyncMock()
    multiple = AsyncMock()
    monkeypatch.setattr(probe, "load_config", lambda: {"api_key": "fixture"})
    monkeypatch.setattr(probe, "get_provider", factory)
    monkeypatch.setattr(probe, "probe_provider", basic)
    monkeypatch.setattr(probe, "probe_historical_depth", depth)
    monkeypatch.setattr(probe, "probe_multiple_symbols", multiple)

    await probe.main(["--provider", "finnhub", "--symbol", "XYZ", "--depth-test", "--multi-symbol"])

    factory.assert_called_once_with("finnhub", "fixture", None)
    basic.assert_awaited_once_with(provider, "XYZ")
    depth.assert_awaited_once_with(provider, "XYZ")
    multiple.assert_awaited_once_with(provider)


@pytest.mark.asyncio
async def test_alpaca_provider_normalizes_mocked_paginated_minute_bars(monkeypatch):
    from tradfi_data import get_provider

    provider = get_provider("alpaca", "fixture-key", "fixture-secret")
    requested = []

    class Response:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        async def json(self):
            return self.payload

    def get(url, *, params, headers):
        requested.append((url, dict(params), dict(headers)))
        minute = "01" if "page_token" in params else "00"
        return Response({
            "bars": [{"t": f"2026-01-05T15:{minute}:00Z", "o": 10, "h": 12, "l": 9, "c": 11, "v": 5}],
            "next_page_token": None if "page_token" in params else "second",
        })

    monkeypatch.setattr(provider, "_session", Mock(get=get))
    rows = await provider.fetch_1m_candles("XYZ", 1767625200000, 1767625320000)
    assert [row.timestamp_ms for row in rows] == [1767625200000, 1767625260000]
    assert [row.close for row in rows] == [11, 11]
    assert requested[0][0].endswith("/XYZ/bars")
    assert requested[0][1]["timeframe"] == "1Min"
    assert "page_token" not in requested[0][1]
    assert requested[1][1]["page_token"] == "second"
    assert requested[0][2]["APCA-API-KEY-ID"] == "fixture-key"
