import asyncio

import pytest

from market_snapshot import MarketSnapshotProvider


@pytest.mark.asyncio
async def test_market_snapshot_provider_fetches_bulk_tickers_and_caches():
    calls = {"fetch": 0}
    cache_sink = []

    async def fetch_tickers():
        calls["fetch"] += 1
        return {
            "BTC/USDT:USDT": {"bid": 99.0, "ask": 101.0, "last": 100.0},
            "ETH/USDT:USDT": {"bid": 199.0, "ask": 201.0, "last": 200.0},
        }

    provider = MarketSnapshotProvider(
        exchange_name="bybit",
        fetch_tickers=fetch_tickers,
        cache_sink=lambda symbol, price, ts: cache_sink.append((symbol, price, ts)),
    )

    first = await provider.get_snapshots(["BTC/USDT:USDT", "ETH/USDT:USDT"], max_age_ms=60_000)
    second = await provider.get_snapshots(["BTC/USDT:USDT"], max_age_ms=60_000)

    assert calls["fetch"] == 1
    assert first["BTC/USDT:USDT"].bid == 99.0
    assert first["BTC/USDT:USDT"].ask == 101.0
    assert first["BTC/USDT:USDT"].last == 100.0
    assert second["BTC/USDT:USDT"].last == 100.0
    assert cache_sink[0][0] == "BTC/USDT:USDT"
    assert cache_sink[0][1] == 100.0


@pytest.mark.asyncio
async def test_market_snapshot_provider_caches_all_bulk_ticker_results():
    calls = {"fetch": 0}

    async def fetch_tickers():
        calls["fetch"] += 1
        return {
            "BTC/USDT:USDT": {"bid": 99.0, "ask": 101.0, "last": 100.0},
            "ETH/USDT:USDT": {"bid": 199.0, "ask": 201.0, "last": 200.0},
        }

    provider = MarketSnapshotProvider(exchange_name="bybit", fetch_tickers=fetch_tickers)

    first = await provider.get_snapshots(["BTC/USDT:USDT"], max_age_ms=60_000)
    second = await provider.get_snapshots(["ETH/USDT:USDT"], max_age_ms=60_000)

    assert calls["fetch"] == 1
    assert first["BTC/USDT:USDT"].last == 100.0
    assert second["ETH/USDT:USDT"].last == 200.0


@pytest.mark.asyncio
async def test_market_snapshot_provider_coalesces_concurrent_bulk_fetches():
    calls = {"fetch": 0}
    release = asyncio.Event()

    async def fetch_tickers():
        calls["fetch"] += 1
        await release.wait()
        return {
            "BTC/USDT:USDT": {"bid": 99.0, "ask": 101.0, "last": 100.0},
            "ETH/USDT:USDT": {"bid": 199.0, "ask": 201.0, "last": 200.0},
        }

    provider = MarketSnapshotProvider(exchange_name="bybit", fetch_tickers=fetch_tickers)

    task_a = asyncio.create_task(provider.get_snapshots(["BTC/USDT:USDT"], max_age_ms=60_000))
    task_b = asyncio.create_task(provider.get_snapshots(["ETH/USDT:USDT"], max_age_ms=60_000))
    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(task_a, task_b)

    assert calls["fetch"] == 1
    assert first["BTC/USDT:USDT"].last == 100.0
    assert second["ETH/USDT:USDT"].last == 200.0


@pytest.mark.asyncio
async def test_market_snapshot_provider_fills_missing_bid_ask_from_last():
    async def fetch_tickers():
        return {"HYPE/USDC:USDC": {"last": 42.0, "bid": None, "ask": None}}

    provider = MarketSnapshotProvider(exchange_name="hyperliquid", fetch_tickers=fetch_tickers)

    out = await provider.get_snapshots(["HYPE/USDC:USDC"], max_age_ms=60_000)

    assert out["HYPE/USDC:USDC"].bid == 42.0
    assert out["HYPE/USDC:USDC"].ask == 42.0
    assert out["HYPE/USDC:USDC"].last == 42.0


@pytest.mark.asyncio
async def test_market_snapshot_provider_returns_partial_on_fetch_failure():
    calls = {"fetch": 0}

    async def fetch_tickers():
        calls["fetch"] += 1
        if calls["fetch"] == 1:
            return {"BTC/USDT:USDT": {"bid": 99.0, "ask": 101.0, "last": 100.0}}
        raise RuntimeError("rate limited")

    provider = MarketSnapshotProvider(exchange_name="bybit", fetch_tickers=fetch_tickers)

    first = await provider.get_snapshots(["BTC/USDT:USDT"], max_age_ms=60_000)
    second = await provider.get_snapshots(["BTC/USDT:USDT", "ETH/USDT:USDT"], max_age_ms=60_000)

    assert first["BTC/USDT:USDT"].last == 100.0
    assert second["BTC/USDT:USDT"].last == 100.0
    assert "ETH/USDT:USDT" not in second
