import asyncio
import logging

import pytest

from market_snapshot import MarketSnapshotProvider
from live.market_snapshot import MarketSnapshotUnavailable


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
async def test_cancelled_waiter_does_not_cancel_shared_quote_or_other_waiter(strategy):
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0
    async def fetch(*args):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {'A': {'bid': 99., 'ask': 101., 'last': 100.}}
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    protection = asyncio.create_task(provider.get_snapshots(['A']))
    ordinary = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    protection.cancel()
    with pytest.raises(asyncio.CancelledError):
        await protection
    release.set()
    assert (await ordinary)['A'].last == 100.
    assert calls == 1
    assert provider._fetch_task is None and not provider._symbol_fetch_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
@pytest.mark.parametrize('outcome', ['success', 'failure', 'shutdown'])
async def test_abandoned_shared_quote_is_owned_until_completion_or_shutdown(strategy, outcome):
    started, release = asyncio.Event(), asyncio.Event()
    async def fetch(*args):
        started.set()
        await release.wait()
        if outcome == 'failure':
            raise OSError('synthetic offline failure')
        return {'A': {'bid': 99., 'ask': 101., 'last': 100.}}
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    waiter = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    shared = provider._fetch_task or next(iter(provider._symbol_fetch_tasks.values()))
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not shared.done()
    if outcome == 'shutdown':
        provider.cancel_pending()
    else:
        release.set()
    await asyncio.wait({shared})
    await asyncio.sleep(0)  # provider completion callback, even without a waiter
    assert provider._fetch_task is None and not provider._symbol_fetch_tasks
    assert shared.cancelled() == (outcome == 'shutdown')
    assert not shared._log_traceback  # an abandoned failure was retrieved


@pytest.mark.asyncio
async def test_revised_quote_deadline_cannot_cancel_ordinary_shared_request():
    from types import SimpleNamespace
    from time import monotonic
    from live.hsl_revised_live import Owner
    started, release = asyncio.Event(), asyncio.Event()
    async def fetch():
        started.set()
        await release.wait()
        return {'A': {'bid': 99., 'ask': 101., 'last': 100.}}
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch)
    owner = Owner(SimpleNamespace(_get_orchestrator_market_snapshots=provider.get_snapshots))
    assert await owner.acquire_quotes({'A'}) == {}
    ordinary = asyncio.create_task(provider.get_snapshots(['A']))
    await asyncio.sleep(0)
    owner._quote_started['A'] = monotonic() - 6.
    assert await owner.acquire_quotes({'A'}) == {}
    await asyncio.sleep(0)
    assert not ordinary.done()
    release.set()
    assert (await ordinary)['A'].last == 100.
    assert (await owner.acquire_quotes({'A'}))['A'].last == 100.
    owner.cancel_inputs()


@pytest.mark.asyncio
async def test_bot_shutdown_cancels_provider_owned_requests():
    from types import SimpleNamespace
    from passivbot import Passivbot
    started = asyncio.Event()
    async def fetch():
        started.set()
        await asyncio.Event().wait()
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch)
    waiter = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    from unittest.mock import AsyncMock
    bot = Passivbot.__new__(Passivbot)
    bot.market_snapshot_provider = provider
    bot.cca, bot.ccp = SimpleNamespace(close=AsyncMock()), None
    bot._close_live_event_pipeline = lambda **kwargs: True
    await bot.close()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert provider._fetch_task is None


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


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['primary', 'missing_symbol'])
@pytest.mark.parametrize('kind', ['AuthenticationError', 'BadRequest', 'NotSupported', 'ValueError', 'TypeError', 'KeyError'])
async def test_market_snapshot_preserves_permanent_connector_errors(path, kind, caplog):
    from ccxt.base import errors
    import builtins
    error_type = getattr(builtins, kind, None) or getattr(errors, kind)
    original = error_type('api_key=private')
    async def fail(*args):
        raise original
    async def empty():
        return {}
    provider = MarketSnapshotProvider(
        exchange_name='bybit', fetch_tickers=fail if path == 'primary' else empty,
        fetch_tickers_for_symbols=fail,
    )
    with caplog.at_level(logging.WARNING):
        with pytest.raises(type(original)) as caught:
            await provider.get_snapshots(['A'])
    assert caught.value is original
    assert 'api_key=private' not in caplog.text


@pytest.mark.asyncio
async def test_crossed_quotes_are_unavailable_and_recover_on_next_fetch():
    from live.market_snapshot import MarketSnapshotUnavailable
    book = {"bid": 101.0, "ask": 100.0, "last": 100.5}
    async def fetch():
        return {"BTC/USDT:USDT": dict(book)}
    provider = MarketSnapshotProvider(exchange_name="fake", fetch_tickers=fetch)
    with pytest.raises(MarketSnapshotUnavailable):
        await provider.get_snapshots(["BTC/USDT:USDT"])
    assert provider.get_cached("BTC/USDT:USDT", now_ms=0, max_age_ms=10_000) is None
    book["ask"] = 102.0
    snapshots = await provider.get_snapshots(["BTC/USDT:USDT"])
    assert snapshots["BTC/USDT:USDT"].is_valid()


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
@pytest.mark.parametrize('error_name', ['AuthenticationError', 'BadRequest', 'ValueError', 'RequestTimeout'])
async def test_abandoned_fetch_preserves_fatal_failure_for_next_reader(strategy, error_name):
    from ccxt.base.errors import AuthenticationError, BadRequest, RequestTimeout
    from market_snapshot import MarketSnapshot
    errors = {e.__name__: e for e in (AuthenticationError, BadRequest, ValueError, RequestTimeout)}
    failure = errors[error_name]('synthetic quote failure')
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def fetch(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            raise failure
        return {'A': {'bid': 99., 'ask': 101., 'last': 100.}}

    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    waiter = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    shared = provider.pending_tasks()[0]
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await asyncio.wait({shared})
    await asyncio.sleep(0)
    if error_name != 'RequestTimeout':
        # Even a different, cached symbol must not bypass late fatal delivery.
        from utils import utc_ms
        provider._cache['B'] = MarketSnapshot('B', 99., 101., 100., utc_ms(), 'test')
        with pytest.raises(errors[error_name]) as caught:
            await provider.get_snapshots(['B'])
        assert caught.value is failure
        assert calls == 1
    assert (await provider.get_snapshots(['A']))['A'].last == 100.
    assert calls == 2
    assert not shared._log_traceback


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
async def test_failure_delivered_to_active_reader_is_not_replayed(strategy):
    calls = 0
    async def fetch(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError('synthetic active failure')
        return {'A': {'bid': 99., 'ask': 101., 'last': 100.}}
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    with pytest.raises(ValueError):
        await provider.get_snapshots(['A'])
    assert (await provider.get_snapshots(['A']))['A'].last == 100.
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
@pytest.mark.parametrize('cleanup', ['restart', 'shutdown', 'close', 'bitunix_close', 'shutdown_bot', 'bitunix_shutdown_bot'])
async def test_bot_cleanup_awaits_shared_quote_cleanup_before_client_close(strategy, cleanup):
    from passivbot import Passivbot
    from unittest.mock import AsyncMock
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []
    async def fetch(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()  # asynchronous connector teardown
            calls.append('quote_stopped')
            raise
    class Client:
        async def close(self):
            assert calls == ['quote_stopped']
            calls.append('client_closed')
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    waiter = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    from exchanges.bitunix import BitunixBot
    cls = BitunixBot if cleanup.startswith('bitunix') else Passivbot
    bot = cls.__new__(cls)
    bot.market_snapshot_provider = provider
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp, bot.cca = None, Client()
    bot.monitor_publisher = None
    bot._close_live_event_pipeline = lambda **kwargs: True
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    from passivbot import shutdown_bot
    if cleanup == 'restart':
        operation = bot.cleanup_for_restart()
    elif cleanup == 'shutdown':
        operation = bot.shutdown_gracefully()
    elif cleanup.endswith('shutdown_bot'):
        operation = shutdown_bot(bot)
    else:
        operation = bot.close()
    task = asyncio.create_task(operation)
    await cancelled.wait()
    assert not task.done() and calls == []
    release.set()
    await task
    assert calls == ['quote_stopped', 'client_closed']
    assert provider.pending_tasks() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
@pytest.mark.parametrize('malformed', [None, [], 42, 'invalid'])
async def test_abandoned_malformed_result_is_validated_by_shared_owner(strategy, malformed):
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0
    async def fetch(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            return malformed
        return {'A': {'bid': 99., 'ask': 101., 'last': 100.}}
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    waiter = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    shared = provider.pending_tasks()[0]
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await asyncio.wait({shared})
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match='returned non-dict'):
        await provider.get_snapshots(['A'])
    assert calls == 1
    assert (await provider.get_snapshots(['A']))['A'].last == 100.
    assert calls == 2


@pytest.mark.asyncio
async def test_shared_quote_cleanup_repeated_cancellation_waits_then_is_bounded():
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def fetch():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await asyncio.Event().wait()
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch)
    reader = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    shared = provider.pending_tasks()[0]
    provider.cancel_pending()
    await cancelled.wait()
    provider.cancel_pending()
    assert shared.cancelling() == 1  # don't interrupt asynchronous cleanup twice
    await asyncio.wait_for(provider.wait_pending(timeout_seconds=0), timeout=1.5)
    with pytest.raises(asyncio.CancelledError):
        await reader
    assert shared.done() and provider.pending_tasks() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize('caller', ['shutdown_bot', 'contract_capture'])
async def test_outer_close_deadline_allows_slow_provider_cleanup(caller, monkeypatch):
    from passivbot import Passivbot, shutdown_bot, BOT_CLOSE_TIMEOUT_SECONDS
    import ccxt_contracts
    from live.market_snapshot import SHARED_QUOTE_CLEANUP_SECONDS, SHARED_QUOTE_CANCEL_GRACE_SECONDS
    assert BOT_CLOSE_TIMEOUT_SECONDS == SHARED_QUOTE_CLEANUP_SECONDS + SHARED_QUOTE_CANCEL_GRACE_SECONDS + 3.0
    started = asyncio.Event()
    calls = []
    async def fetch():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(3.2)  # outlast the former outer three-second limit
            calls.append('quote_stopped')
            raise
    class Client:
        async def close(self):
            assert calls == ['quote_stopped']
            calls.append('client_closed')
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch)
    reader = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    bot = Passivbot.__new__(Passivbot)
    bot.market_snapshot_provider = provider
    bot.exchange = 'fake'
    bot.cca, bot.ccp = Client(), None
    bot._close_live_event_pipeline = lambda **kwargs: True
    if caller == 'shutdown_bot':
        await shutdown_bot(bot)
    else:
        monkeypatch.setattr(ccxt_contracts, 'prepare_live_config_for_user', lambda _: {})
        monkeypatch.setattr(ccxt_contracts, 'setup_bot', lambda _: bot)
        await ccxt_contracts.capture_contract_snapshot(user='synthetic', sections=())
    assert calls == ['quote_stopped', 'client_closed']
    assert provider.pending_tasks() == ()
    with pytest.raises(asyncio.CancelledError):
        await reader


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
async def test_replacing_maintainers_preserves_shared_quote_readers(strategy):
    from passivbot import Passivbot
    started, release = asyncio.Event(), asyncio.Event()
    async def fetch(*args):
        started.set()
        await release.wait()
        return {'A': {'bid': 99., 'ask': 101., 'last': 100.}}
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    ordinary = asyncio.create_task(provider.get_snapshots(['A']))
    protective = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    shared = provider.pending_tasks()[0]
    bot = Passivbot.__new__(Passivbot)
    bot.market_snapshot_provider = provider
    bot.ws_enabled = False
    bot.maintain_hourly_cycle = lambda: asyncio.Event().wait()
    old = asyncio.create_task(asyncio.Event().wait())
    bot.maintainers = {'maintain_hourly_cycle': old}
    try:
        await bot.start_data_maintainers()
        await asyncio.sleep(0)
        assert old.cancelled()
        assert not shared.cancelling()
        assert not ordinary.done() and not protective.done()
        release.set()
        for reader in (ordinary, protective):
            assert (await reader)['A'].last == 100.
    finally:
        bot.stop_data_maintainers(verbose=False)
        for task in bot.maintainers.values():
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.parametrize('bot_type', ['base', 'bitunix'])
@pytest.mark.parametrize('failure', ['rest', 'stream', 'rest_cancelled'])
async def test_close_attempts_all_resources_after_client_failure(bot_type, failure):
    from passivbot import Passivbot
    from exchanges.bitunix import BitunixBot
    calls = []
    class Client:
        def __init__(self, name):
            self.name = name
        async def close(self):
            calls.append(self.name)
            if failure == self.name:
                raise RuntimeError('test close failure')
            if failure == 'rest_cancelled' and self.name == 'rest':
                raise asyncio.CancelledError()
    cls = Passivbot if bot_type == 'base' else BitunixBot
    bot = cls.__new__(cls)
    bot.cca, bot.ccp = Client('rest'), Client('stream')
    bot._close_live_event_pipeline = lambda **kwargs: calls.append('pipeline')
    error = asyncio.CancelledError if failure == 'rest_cancelled' else RuntimeError
    with pytest.raises(error):
        await bot.close()
    assert calls == ['rest', 'stream', 'pipeline']


@pytest.mark.asyncio
@pytest.mark.parametrize('first', ['error', 'cancelled'])
async def test_close_preserves_first_failure_when_later_cleanup_also_fails(first, caplog):
    from passivbot import Passivbot
    calls = []
    primary = ValueError('first failure') if first == 'error' else asyncio.CancelledError()
    class Client:
        def __init__(self, name, error):
            self.name, self.error = name, error
        async def close(self):
            calls.append(self.name)
            raise self.error
    bot = Passivbot.__new__(Passivbot)
    bot.cca, bot.ccp = Client('rest', primary), Client('stream', RuntimeError('stream-secret'))
    def pipeline(**kwargs):
        calls.append('pipeline')
        raise OSError('pipeline-secret')
    bot._close_live_event_pipeline = pipeline
    with pytest.raises(type(primary)) as raised:
        await bot.close()
    assert raised.value is primary
    assert calls == ['rest', 'stream', 'pipeline']
    assert caplog.text.count('additional close failure') == 2
    assert 'stream-secret' not in caplog.text and 'pipeline-secret' not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
@pytest.mark.parametrize('release_cleanup', [True, False])
async def test_cancelled_close_finishes_one_bounded_quote_wait(strategy, release_cleanup):
    from passivbot import Passivbot
    from functools import partial
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []
    async def fetch(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            try:
                await release.wait()
            finally:
                calls.append('quote_stopped')
            raise
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    provider.wait_pending = partial(provider.wait_pending, timeout_seconds=.05)
    reader = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    class Client:
        def __init__(self, name):
            self.name = name
        async def close(self):
            assert calls[0] == 'quote_stopped'
            calls.append(self.name)
    bot = Passivbot.__new__(Passivbot)
    bot.market_snapshot_provider = provider
    bot.cca, bot.ccp = Client('rest'), Client('stream')
    bot._close_live_event_pipeline = lambda **kwargs: calls.append('pipeline')
    closer = asyncio.create_task(bot.close())
    await cancelled.wait()
    for _ in range(2):
        closer.cancel()
        await asyncio.sleep(0)
        assert not closer.done() and calls == []
    if release_cleanup:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closer, 1.)
    with pytest.raises(asyncio.CancelledError):
        await reader
    assert calls == ['quote_stopped', 'rest', 'stream', 'pipeline']
    assert provider.pending_tasks() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
async def test_shutdown_rejects_quote_request_from_late_execution_cycle(strategy):
    from passivbot import Passivbot
    from unittest.mock import AsyncMock
    permit = asyncio.Event()
    calls = []
    async def fetch(*args):
        calls.append('fetch')
        return {'A': {'bid': 99., 'ask': 101., 'last': 100.}}
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    async def execution():
        await permit.wait()  # already entered cycle before shutdown
        with pytest.raises(MarketSnapshotUnavailable, match='shutting down'):
            await provider.get_snapshots(['A'])
        calls.append('execution_stopped')
    class Client:
        async def close(self):
            assert calls == ['execution_stopped']
            calls.append('client_closed')
    bot = Passivbot.__new__(Passivbot)
    bot.market_snapshot_provider = provider
    bot.maintainers, bot.WS_ohlcvs_1m_tasks = {}, {}
    bot.cca, bot.ccp = Client(), None
    bot.monitor_publisher = None
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot._close_live_event_pipeline = lambda **kwargs: True
    bot._emit_shutdown_stage = lambda stage, **kwargs: permit.set() if stage == 'execution_loop_waiting' else None
    bot._execution_loop_task = asyncio.create_task(execution())
    await bot.shutdown_gracefully()
    await bot._execution_loop_task
    assert calls == ['execution_stopped', 'client_closed']
    assert provider.pending_tasks() == ()


@pytest.mark.asyncio
async def test_shutdown_blocks_cache_reads_and_retry_after_suppressed_cancellation():
    started = asyncio.Event()
    calls = []
    async def bulk():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return {}  # a connector can finish normally after cancellation
    async def symbols(names):
        calls.append(names)
        return {'A': {'bid': 99., 'ask': 101., 'last': 100.}}
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=bulk,
        fetch_tickers_for_symbols=symbols)
    reader = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    provider.begin_shutdown()
    with pytest.raises(MarketSnapshotUnavailable):
        await reader
    await provider.wait_pending()
    assert not calls and provider.pending_tasks() == ()
    provider._cache['A'] = provider._snapshot_from_ticker('A', {'bid': 99., 'ask': 101., 'last': 100.},
        fetched_ms=10**15)
    with pytest.raises(MarketSnapshotUnavailable, match='shutting down'):
        await provider.get_snapshots(['A'])


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
@pytest.mark.parametrize('completed_before_close', [False, True])
async def test_close_reports_abandoned_quote_cleanup_failure(strategy, completed_before_close, caplog):
    from passivbot import Passivbot
    started = asyncio.Event()
    primary = RuntimeError('connector-secret')
    calls = []
    async def fetch(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise primary
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    reader = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    if completed_before_close:
        owned = provider.pending_tasks()
        provider.begin_shutdown()
        await asyncio.wait(owned)
    class Client:
        async def close(self):
            calls.append('client_closed')
    bot = Passivbot.__new__(Passivbot)
    bot.market_snapshot_provider = provider
    bot.cca, bot.ccp = Client(), None
    bot._close_live_event_pipeline = lambda **kwargs: calls.append('pipeline')
    with pytest.raises(RuntimeError) as raised:
        await bot.close()
    assert raised.value is primary
    assert calls == ['client_closed', 'pipeline']
    assert provider.pending_tasks() == ()
    assert provider._pending_failure is None
    assert 'report_cleanup_failure' in caplog.text and 'connector-secret' not in caplog.text
    await provider.wait_pending()  # delivered cleanup errors are not replayed


@pytest.mark.asyncio
async def test_shutdown_reports_failure_after_bounded_abandonment(monkeypatch, caplog):
    import live.market_snapshot as snapshots
    monkeypatch.setattr(snapshots, 'SHARED_QUOTE_CANCEL_GRACE_SECONDS', .01)
    started, cancelled_twice, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def fetch():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled_twice.set()
                await release.wait()
                raise ValueError('late-secret')
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch)
    reader = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    provider.begin_shutdown()
    await asyncio.sleep(0)
    await provider.wait_pending(timeout_seconds=0)
    assert cancelled_twice.is_set() and len(provider.pending_tasks()) == 1
    owned = provider.pending_tasks()
    release.set()
    await asyncio.wait(owned)
    with pytest.raises(ValueError):
        await provider.wait_pending()
    assert 'report_cleanup_failure' in caplog.text and 'late-secret' not in caplog.text
    await provider.wait_pending()


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
@pytest.mark.parametrize('cached', [False, True])
async def test_completed_fatal_precedes_callback_cache_and_replacement(strategy, cached):
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0
    primary = ValueError('private-connector-detail')
    async def fetch(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            raise primary
        return {'A': {'bid': 99., 'ask': 101., 'last': 100.}}
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    abandoned = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    owned = provider.pending_tasks()[0]
    if cached:
        provider._cache['A'] = provider._snapshot_from_ticker('A',
            {'bid': 99., 'ask': 101., 'last': 100.}, fetched_ms=10**15)
    # The failing fetch runs first; the new reader then runs before its callbacks.
    release.set()
    async def read_before_callback():
        assert owned.done() and provider._pending_failure is None
        return await provider.get_snapshots(['A'])
    reader = asyncio.create_task(read_before_callback())
    with pytest.raises(ValueError) as raised:
        await reader
    assert raised.value is primary and calls == 1
    await asyncio.sleep(0)
    assert provider._pending_failure is None  # callback cannot replay delivered failure
    assert (await provider.get_snapshots(['A']))['A'].last == 100.
    assert calls == (1 if cached else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize('lifecycle', ['cleanup_for_restart', 'shutdown_gracefully'])
@pytest.mark.parametrize('strategy', ['bulk', 'symbols'])
async def test_lifecycle_reports_failure_retained_before_shutdown(lifecycle, strategy, caplog):
    from passivbot import Passivbot
    from unittest.mock import AsyncMock
    started, release = asyncio.Event(), asyncio.Event()
    primary = ValueError('private-failure-text')
    async def fetch(*args):
        started.set()
        await release.wait()
        raise primary
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch,
        fetch_tickers_for_symbols=fetch, ticker_strategy=strategy)
    reader = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    owned = provider.pending_tasks()
    release.set()
    await asyncio.wait(owned)
    assert provider._pending_failure is primary and provider.pending_tasks() == ()
    calls = []
    class Client:
        async def close(self):
            calls.append('client')
    bot = Passivbot.__new__(Passivbot)
    bot.market_snapshot_provider = provider
    bot.maintainers, bot.WS_ohlcvs_1m_tasks = {}, {}
    bot.cca, bot.ccp = Client(), None
    bot.monitor_publisher = None
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot._close_live_event_pipeline = lambda **kwargs: True
    bot._emit_shutdown_stage = lambda *args, **kwargs: None
    await getattr(bot, lifecycle)()
    assert calls == ['client'] and provider._pending_failure is None
    assert 'quote cleanup failed' in caplog.text and 'ValueError' in caplog.text
    assert 'private-failure-text' not in caplog.text
    await provider.wait_pending()


@pytest.mark.asyncio
@pytest.mark.parametrize('external_cancel', [False, True])
async def test_close_bounds_all_clients_even_if_they_suppress_cancellation(external_cancel, caplog):
    from passivbot import Passivbot
    started, release = asyncio.Event(), asyncio.Event()
    calls, owned = [], []
    class Client:
        def __init__(self, name):
            self.name = name
        async def close(self):
            owned.append(asyncio.current_task())
            calls.append(self.name)
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
                raise ValueError('late-private-text')
    bot = Passivbot.__new__(Passivbot)
    bot.cca, bot.ccp = Client('rest'), Client('stream')
    bot._close_live_event_pipeline = lambda **kwargs: calls.append('pipeline')
    t0 = asyncio.get_running_loop().time()
    closer = asyncio.create_task(bot.close())
    await started.wait()
    if external_cancel:
        closer.cancel()
    try:
        done, pending = await asyncio.wait({closer}, timeout=1.5)
        assert not pending, 'cleanup swallowed cancellation and hung on another client'
        with pytest.raises(asyncio.CancelledError if external_cancel else asyncio.TimeoutError):
            closer.result()
        assert asyncio.get_running_loop().time() - t0 < 1.5
        assert calls == ['rest', 'stream', 'pipeline']
        assert len(owned) == 2 and all(t.cancelling() for t in owned)
    finally:
        release.set()
        await asyncio.gather(*owned, return_exceptions=True)
        await asyncio.sleep(0)
    assert 'late-private-text' not in caplog.text


@pytest.mark.asyncio
async def test_graceful_shutdown_bounds_quote_that_ignores_repeated_cancellation(caplog):
    from passivbot import Passivbot
    from unittest.mock import AsyncMock
    started, release = asyncio.Event(), asyncio.Event()
    async def fetch():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        raise ValueError('private-late-quote')
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch)
    reader = asyncio.create_task(provider.get_snapshots(['A']))
    await started.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    owned = provider.pending_tasks()
    calls = []
    class Client:
        async def close(self):
            calls.append('client')
    bot = Passivbot.__new__(Passivbot)
    bot.market_snapshot_provider = provider
    bot.maintainers, bot.WS_ohlcvs_1m_tasks = {}, {}
    bot.cca, bot.ccp = Client(), None
    bot.monitor_publisher = None
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._shutdown_maintainer_grace_seconds = .01
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot._close_live_event_pipeline = lambda **kwargs: True
    bot._emit_shutdown_stage = lambda *args, **kwargs: None
    closer = asyncio.create_task(bot.shutdown_gracefully())
    try:
        _, pending = await asyncio.wait({closer}, timeout=1.5)
        assert not pending, 'graceful shutdown exceeded its cancellation grace'
        closer.result()
        assert calls == ['client']
    finally:
        release.set()
        await asyncio.gather(*owned, return_exceptions=True)
        await closer
    assert 'private-late-quote' not in caplog.text
    assert 'report_cleanup_failure' in caplog.text
