from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

import fill_events_manager as fem


START_MS = 1_700_000_000_000


class GateioPages:
    def __init__(self, mode):
        self.mode = mode
        self.calls = 0

    async def private_futures_get_settle_my_trades_timerange(self, params):
        self.calls += 1
        if self.mode == "rate_limit":
            raise fem.RateLimitExceeded("rate limited")
        if self.mode == "disguised_rate_limit":
            raise RuntimeError("TOO_MANY_REQUESTS")
        if self.mode == "empty":
            return []
        count = params["limit"]
        if self.calls == 400 and self.mode in {"terminal_empty", "terminal_short"}:
            count = 0 if self.mode == "terminal_empty" else 1
        return [
            {
                "trade_id": str(params["offset"] + index),
                "create_time": START_MS / 1000 + self.calls,
                "contract": "BTC_USDT",
                "size": 1,
                "price": 100,
                "fee": 0.01,
            }
            for index in range(count)
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["rate_limit", "disguised_rate_limit", "full_pages"])
async def test_gateio_incomplete_lookback_preserves_coverage_and_checkpoint(
    tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(fem.asyncio, "sleep", AsyncMock())
    api = GateioPages(mode)
    manager = fem.FillEventsManager(
        exchange="gateio",
        user="test",
        fetcher=fem.GateioFetcher(api, trade_limit=2),
        cache_path=tmp_path,
    )
    metadata = manager.cache.load_metadata()
    metadata.update(last_refresh_ms=START_MS + 1000, covered_start_ms=START_MS + 1000)
    manager.cache.save_metadata()
    before = deepcopy(metadata)

    with pytest.raises(RuntimeError, match="incomplete history"):
        await manager.refresh_for_lookback(START_MS)

    assert api.calls == 400
    assert manager.cache.load_metadata() == before
    assert manager.get_events() == []
    assert not manager.get_coverage_status(start_ms=START_MS)["ready"]
    assert not list(tmp_path.glob("????-??-??.json"))

    # An actual completed retry proves the window, including a legitimate empty account.
    api.mode = "empty"
    await manager.refresh_for_lookback(START_MS)
    assert manager.get_coverage_status(start_ms=START_MS)["ready"]
    assert manager.cache.load_metadata()["last_refresh_ms"] > before["last_refresh_ms"]


@pytest.mark.asyncio
async def test_gateio_incomplete_bounded_refresh_records_retryable_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(fem.asyncio, "sleep", AsyncMock())
    api = GateioPages("full_pages")
    manager = fem.FillEventsManager(
        exchange="gateio",
        user="test",
        fetcher=fem.GateioFetcher(api, trade_limit=2),
        cache_path=tmp_path,
    )
    metadata = manager.cache.load_metadata()
    metadata.update(last_refresh_ms=START_MS, covered_start_ms=START_MS)
    manager.cache.save_metadata()
    end_ms = START_MS + 1_000_000

    with pytest.raises(RuntimeError, match="incomplete history"):
        await manager.refresh(start_ms=START_MS, end_ms=end_ms)

    assert manager.cache.load_metadata()["last_refresh_ms"] == START_MS
    assert manager.get_events() == []
    assert not manager.get_coverage_status(start_ms=START_MS, end_ms=end_ms)["ready"]
    assert manager.cache.get_known_gaps()[0]["reason"] == fem.GAP_REASON_FETCH_FAILED

    api.mode = "empty"
    await manager.refresh_range(start_ms=START_MS, end_ms=end_ms)
    assert manager.cache.get_known_gaps() == []
    assert manager.get_coverage_status(start_ms=START_MS, end_ms=end_ms)["ready"]
    assert manager.cache.load_metadata()["last_refresh_ms"] == START_MS


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_rows", [0, 1])
async def test_gateio_accepts_terminal_page_on_last_allowed_request(monkeypatch, terminal_rows):
    monkeypatch.setattr(fem.asyncio, "sleep", AsyncMock())
    api = GateioPages("terminal_short" if terminal_rows else "terminal_empty")
    fetcher = fem.GateioFetcher(api, trade_limit=2)

    trades = await fetcher._fetch_trades(START_MS, START_MS + 1_000_000)

    assert api.calls == 400
    assert len(trades) == 798 + terminal_rows
    assert len({trade["id"] for trade in trades}) == len(trades)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["trades", "positions_history"])
@pytest.mark.parametrize("empty_windows", [False, True])
@pytest.mark.parametrize("complete", [False, True])
async def test_kucoin_requires_requested_range_completion(endpoint, empty_windows, complete):
    class KucoinPages:
        calls = 0

        async def fetch_my_trades(self, params):
            self.calls += 1
            if empty_windows:
                return []
            return [
                {
                    "id": str(self.calls),
                    "timestamp": params["startAt"] + 1,
                    "symbol": "BTC/USDT:USDT",
                    "side": "buy",
                    "amount": 1,
                    "price": 100,
                    "fee": {"currency": "USDT", "cost": 0.01},
                }
            ]

        async def fetch_positions_history(self, params):
            self.calls += 1
            if empty_windows:
                return []
            return [{"id": str(self.calls), "lastUpdateTimestamp": params["from"] + 1}]

    api = KucoinPages()
    fetcher = fem.KucoinFetcher(api)
    # Nonempty pages advance two milliseconds; empty pages traverse a full request window.
    step_ms = int(24 * 60 * 60 * 1000 * 0.99) if empty_windows else 2
    end_ms = START_MS + step_ms * (400 if complete else 401)
    fetch = getattr(fetcher, f"_fetch_{endpoint}")

    if complete:
        rows = await fetch(START_MS, end_ms)
        assert len(rows) == (0 if empty_windows else 400)
    else:
        with pytest.raises(RuntimeError, match="incomplete history"):
            await fetch(START_MS, end_ms)
    assert api.calls == 400
