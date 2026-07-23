import asyncio
import logging

import pytest

from market_snapshot import MarketSnapshotProvider


def _unsafe_snapshot_exception(secret: str) -> RuntimeError:
    unsafe_type = type("SnapshotCredentialFailure", (RuntimeError,), {})
    return unsafe_type(secret)


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
async def test_market_snapshot_provider_retries_missing_bulk_symbols_strictly():
    calls = {"bulk": 0, "symbols": []}

    async def fetch_tickers():
        calls["bulk"] += 1
        return {
            "BTC/USDT:USDT": {"bid": 99.0, "ask": 101.0, "last": 100.0},
        }

    async def fetch_tickers_for_symbols(symbols):
        calls["symbols"].append(list(symbols))
        return {
            "ETH/USDT:USDT": {"bid": 199.0, "ask": 201.0, "last": 200.0},
        }

    provider = MarketSnapshotProvider(
        exchange_name="gateio",
        fetch_tickers=fetch_tickers,
        fetch_tickers_for_symbols=fetch_tickers_for_symbols,
    )

    out = await provider.get_snapshots(
        ["BTC/USDT:USDT", "ETH/USDT:USDT"], max_age_ms=60_000
    )

    assert calls["bulk"] == 1
    assert calls["symbols"] == [["ETH/USDT:USDT"]]
    assert out["BTC/USDT:USDT"].source == "fetch_tickers"
    assert out["ETH/USDT:USDT"].source == "fetch_tickers_symbols"
    assert out["ETH/USDT:USDT"].last == 200.0


@pytest.mark.asyncio
async def test_market_snapshot_provider_uses_symbol_strategy():
    calls = {"bulk": 0, "symbols": []}

    async def fetch_tickers():
        calls["bulk"] += 1
        return {}

    async def fetch_tickers_for_symbols(symbols):
        calls["symbols"].append(list(symbols))
        return {
            symbol: {"bid": 99.0, "ask": 101.0, "last": 100.0}
            for symbol in symbols
        }

    provider = MarketSnapshotProvider(
        exchange_name="bitget",
        fetch_tickers=fetch_tickers,
        fetch_tickers_for_symbols=fetch_tickers_for_symbols,
        ticker_strategy="symbols",
    )

    first = await provider.get_snapshots(["BTC/USDC:USDC", "ETH/USDC:USDC"], max_age_ms=60_000)
    second = await provider.get_snapshots(["BTC/USDC:USDC"], max_age_ms=60_000)

    assert calls["bulk"] == 0
    assert calls["symbols"] == [["BTC/USDC:USDC", "ETH/USDC:USDC"]]
    assert first["BTC/USDC:USDC"].source == "fetch_tickers_symbols"
    assert first["ETH/USDC:USDC"].last == 100.0
    assert second["BTC/USDC:USDC"].last == 100.0


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
@pytest.mark.parametrize(
    "ticker",
    [
        {"last": 42.0, "bid": None, "ask": 43.0},
        {"last": 42.0, "bid": 41.0, "ask": None},
        {"last": None, "bid": 41.0, "ask": 43.0},
    ],
)
async def test_market_snapshot_provider_rejects_partial_ticker_fields(ticker):
    async def fetch_tickers():
        return {"HYPE/USDC:USDC": ticker}

    provider = MarketSnapshotProvider(exchange_name="hyperliquid", fetch_tickers=fetch_tickers)

    with pytest.raises(RuntimeError, match="ticker snapshots incomplete"):
        await provider.get_snapshots(["HYPE/USDC:USDC"], max_age_ms=60_000)


@pytest.mark.asyncio
async def test_market_snapshot_provider_raises_on_fetch_failure_for_missing_symbols():
    calls = {"fetch": 0}

    async def fetch_tickers():
        calls["fetch"] += 1
        if calls["fetch"] == 1:
            return {"BTC/USDT:USDT": {"bid": 99.0, "ask": 101.0, "last": 100.0}}
        raise RuntimeError("rate limited")

    provider = MarketSnapshotProvider(exchange_name="bybit", fetch_tickers=fetch_tickers)

    first = await provider.get_snapshots(["BTC/USDT:USDT"], max_age_ms=60_000)

    assert first["BTC/USDT:USDT"].last == 100.0
    with pytest.raises(RuntimeError, match="ticker snapshot fetch failed"):
        await provider.get_snapshots(["BTC/USDT:USDT", "ETH/USDT:USDT"], max_age_ms=60_000)


@pytest.mark.asyncio
async def test_market_snapshot_provider_redacts_primary_fetch_failure_and_preserves_cause(caplog):
    secret = "api_key=provider-secret https://example.invalid/request"
    original = _unsafe_snapshot_exception(secret)

    async def fetch_tickers():
        raise original

    provider = MarketSnapshotProvider(exchange_name="bybit", fetch_tickers=fetch_tickers)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="ticker snapshot fetch failed") as raised:
            await provider.get_snapshots(["BTC/USDT:USDT"], max_age_ms=60_000)

    assert raised.value.__cause__ is original
    assert "error_type=RuntimeError" in caplog.text
    assert "action=propagate" in caplog.text
    assert secret not in caplog.text
    assert type(original).__name__ not in caplog.text


@pytest.mark.asyncio
async def test_market_snapshot_provider_redacts_missing_symbol_retry_failure_and_preserves_cause(
    caplog,
):
    secret = "token=retry-secret https://example.invalid/retry"
    original = _unsafe_snapshot_exception(secret)
    calls = {"symbols": []}

    async def fetch_tickers():
        return {"BTC/USDT:USDT": {"bid": 99.0, "ask": 101.0, "last": 100.0}}

    async def fetch_tickers_for_symbols(symbols):
        calls["symbols"].append(list(symbols))
        raise original

    provider = MarketSnapshotProvider(
        exchange_name="bybit",
        fetch_tickers=fetch_tickers,
        fetch_tickers_for_symbols=fetch_tickers_for_symbols,
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="ticker missing-symbol retry failed") as raised:
            await provider.get_snapshots(
                ["BTC/USDT:USDT", "ETH/USDT:USDT"], max_age_ms=60_000
            )

    assert calls["symbols"] == [["ETH/USDT:USDT"]]
    assert raised.value.__cause__ is original
    assert "error_type=RuntimeError" in caplog.text
    assert "action=propagate" in caplog.text
    assert secret not in caplog.text
    assert type(original).__name__ not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("use_retry", [False, True], ids=["initial", "retry"])
async def test_market_snapshot_provider_cache_sink_failure_is_redacted_and_nonblocking(
    use_retry, caplog
):
    secret = "password=cache-secret https://example.invalid/cache"
    original = _unsafe_snapshot_exception(secret)
    sink_calls = []

    async def fetch_tickers():
        return (
            {"BTC/USDT:USDT": {"bid": 99.0, "ask": 101.0, "last": 100.0}}
            if use_retry
            else {"ETH/USDT:USDT": {"bid": 199.0, "ask": 201.0, "last": 200.0}}
        )

    async def fetch_tickers_for_symbols(symbols):
        assert use_retry
        assert symbols == ["ETH/USDT:USDT"]
        return {"ETH/USDT:USDT": {"bid": 199.0, "ask": 201.0, "last": 200.0}}

    def cache_sink(symbol, price, timestamp):
        sink_calls.append((symbol, price, timestamp))
        raise original

    provider = MarketSnapshotProvider(
        exchange_name="bybit",
        fetch_tickers=fetch_tickers,
        fetch_tickers_for_symbols=fetch_tickers_for_symbols if use_retry else None,
        cache_sink=cache_sink,
    )
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"] if use_retry else ["ETH/USDT:USDT"]

    with caplog.at_level(logging.DEBUG):
        out = await provider.get_snapshots(symbols, max_age_ms=60_000)

    assert set(out) == set(symbols)
    assert [call[0] for call in sink_calls] == symbols
    assert "error_type=RuntimeError" in caplog.text
    assert "action=preserve_snapshot" in caplog.text
    assert secret not in caplog.text
    assert type(original).__name__ not in caplog.text


@pytest.mark.asyncio
async def test_market_snapshot_provider_uses_explicit_ticker_source_label():
    async def fetch_tickers():
        return {
            "HYPE/USDC:USDC": {
                "bid": 42.0,
                "ask": 42.0,
                "last": 42.0,
                "source": "hyperliquid_all_mids",
            }
        }

    provider = MarketSnapshotProvider(exchange_name="hyperliquid", fetch_tickers=fetch_tickers)

    out = await provider.get_snapshots(["HYPE/USDC:USDC"], max_age_ms=60_000)

    assert out["HYPE/USDC:USDC"].bid == 42.0
    assert out["HYPE/USDC:USDC"].source == "hyperliquid_all_mids"
