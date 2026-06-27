import pytest

import tools.probe_ccxt_ticker_endpoints as ticker_endpoint_probe
from tools.probe_ticker_capabilities import (
    create_exchange,
    probe_ticker_capabilities,
    resolve_symbols,
    summarize_order_book,
    summarize_ticker,
    summarize_tickers,
)
from tools.probe_ccxt_ticker_endpoints import (
    probe_exchange_ticker_endpoints,
    select_default_symbols,
    summarize_candle_freshness_probe_collection,
    summarize_candle_freshness_probe_health,
    summarize_endpoint_latency_probe_collection,
    summarize_endpoint_latency_probe_health,
    summarize_fill_history_probe_collection,
    summarize_fill_history_probe_health,
    summarize_my_trades,
    summarize_ohlcvs,
    summarize_rate_limit_probe_collection,
    summarize_rate_limit_probe_health,
    summarize_account_critical_probe_collection,
    summarize_account_critical_probe_health,
    summarize_time_sync_probe_collection,
    summarize_time_sync_probe_health,
)


def test_summarize_ticker_reports_price_field_presence():
    out = summarize_ticker(
        {
            "symbol": "BTC/USDT:USDT",
            "timestamp": 123,
            "datetime": "1970-01-01T00:00:00.123Z",
            "last": 100.0,
            "bid": 99.5,
            "ask": None,
            "info": {"raw": "x"},
        }
    )

    assert out["symbol"] == "BTC/USDT:USDT"
    assert out["has_last"] is True
    assert out["has_bid"] is True
    assert out["has_ask"] is False
    assert "last" in out["top_level_keys"]
    assert out["info_keys"] == ["raw"]


def test_summarize_tickers_reports_requested_symbol_hits():
    out = summarize_tickers(
        {"BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "last": 100.0}},
        ["BTC/USDT:USDT", "ETH/USDT:USDT"],
    )

    assert out["count"] == 1
    assert out["hits"] == {"BTC/USDT:USDT": True, "ETH/USDT:USDT": False}
    assert out["samples"]["BTC/USDT:USDT"]["has_last"] is True


def test_summarize_order_book_reports_top_of_book():
    out = summarize_order_book({"bids": [[99.0, 2.0]], "asks": [[101.0, 3.0]]})

    assert out["has_bid"] is True
    assert out["has_ask"] is True
    assert out["bid"] == 99.0
    assert out["ask"] == 101.0


def test_summarize_ohlcvs_reports_bounded_tail_freshness(monkeypatch):
    monkeypatch.setattr(ticker_endpoint_probe, "utc_ms", lambda: 1_180_000)

    out = summarize_ohlcvs(
        [
            [1_000_000, 9.0, 11.0, 8.0, 10.0, 100.0],
            [1_120_000, 10.0, 12.0, 9.0, 11.0, 101.0],
        ]
    )

    assert out["count"] == 2
    assert out["observed_at_ms"] == 1_180_000
    assert out["last_timestamp"] == 1_120_000
    assert out["last_age_ms"] == 60_000
    assert out["last_age_minutes"] == 1.0
    assert out["last_is_current_incomplete_minute"] is False


@pytest.mark.asyncio
async def test_probe_ticker_capabilities_summarizes_success_and_errors():
    class FakeExchange:
        id = "fake"
        has = {"fetchTicker": True, "fetchTickers": True, "fetchOrderBook": True}

        async def fetch_ticker(self, symbol):
            if symbol == "BAD/USDT:USDT":
                raise RuntimeError("ticker unavailable")
            return {"symbol": symbol, "last": 10.0, "bid": 9.0, "ask": 11.0}

        async def fetch_tickers(self, symbols=None):
            if symbols is None:
                return {
                    "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "last": 10.0},
                    "ETH/USDT:USDT": {"symbol": "ETH/USDT:USDT", "last": 20.0},
                }
            return {symbol: {"symbol": symbol, "last": 10.0} for symbol in symbols}

        async def fetch_order_book(self, symbol, limit=None):
            return {"bids": [[9.0, 1.0]], "asks": [[11.0, 1.0]]}

    out = await probe_ticker_capabilities(
        FakeExchange(),
        ["BTC/USDT:USDT", "BAD/USDT:USDT"],
        probe_all=True,
        probe_order_book=True,
    )

    assert out["exchange"] == "fake"
    assert out["fetch_ticker"]["BTC/USDT:USDT"]["ok"] is True
    assert out["fetch_ticker"]["BAD/USDT:USDT"]["ok"] is False
    assert out["fetch_ticker"]["BAD/USDT:USDT"]["error_type"] == "RuntimeError"
    assert out["fetch_tickers_symbols"]["ok"] is True
    assert out["fetch_tickers_all"]["ok"] is True
    assert out["fetch_order_book"]["BTC/USDT:USDT"]["value"]["bid"] == 9.0


def test_select_default_symbols_prefers_active_linear_swaps_for_quote():
    markets = {
        "BTC/USDT:USDT": {"quote": "USDT", "active": True, "swap": True, "linear": True},
        "ETH/USDT:USDT": {"quote": "USDT", "active": True, "swap": True, "linear": True},
        "BTC/USD:BTC": {"quote": "USD", "active": True, "swap": True, "linear": False},
        "ETH/USD:USD": {"quote": "USD", "active": True, "future": True, "contract": True},
        "COIN/USDT:USDT": {"quote": "USDT", "active": True, "contract": True, "linear": True},
        "OLD/USDT:USDT": {"quote": "USDT", "active": False, "swap": True, "linear": True},
        "SPOT/USDT": {"quote": "USDT", "active": True, "spot": True},
    }

    assert select_default_symbols(markets, quote="USDT", max_symbols=2) == [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
    ]


@pytest.mark.asyncio
async def test_resolve_symbols_requires_active_linear_swaps_for_coins():
    class FakeExchange:
        async def load_markets(self):
            return {
                "ETH/BTC:BTC": {
                    "base": "ETH",
                    "quote": "BTC",
                    "active": True,
                    "swap": True,
                    "linear": False,
                },
                "ETH/USD:USD": {
                    "base": "ETH",
                    "quote": "USD",
                    "active": True,
                    "future": True,
                    "contract": True,
                    "linear": False,
                },
                "ETH/USDT:USDT": {
                    "base": "ETH",
                    "quote": "USDT",
                    "active": True,
                    "swap": True,
                    "linear": True,
                },
            }

    assert await resolve_symbols(FakeExchange(), ["ETH"], quote="USDT") == ["ETH/USDT:USDT"]


@pytest.mark.asyncio
async def test_resolve_symbols_accepts_hyperliquid_hip3_coin_aliases():
    class FakeExchange:
        async def load_markets(self):
            return {
                "XYZ-SP500/USDC:USDC": {
                    "base": "XYZ-SP500",
                    "baseName": "xyz:SP500",
                    "quote": "USDC",
                    "active": True,
                    "swap": True,
                    "linear": True,
                    "info": {"name": "xyz:SP500"},
                },
                "BTC/USDC:USDC": {
                    "base": "BTC",
                    "quote": "USDC",
                    "active": True,
                    "swap": True,
                    "linear": True,
                },
            }

    assert await resolve_symbols(FakeExchange(), ["SP500"], quote="USDC") == [
        "XYZ-SP500/USDC:USDC"
    ]
    assert await resolve_symbols(FakeExchange(), ["xyz:SP500"], quote="USDC") == [
        "XYZ-SP500/USDC:USDC"
    ]
    assert await resolve_symbols(FakeExchange(), ["XYZ-SP500"], quote="USDC") == [
        "XYZ-SP500/USDC:USDC"
    ]


def test_create_exchange_sets_default_type_swap_before_load_markets(monkeypatch):
    captured = {}

    class FakeCCXTExchange:
        def __init__(self, config):
            captured["config"] = config
            self.options = {}

    monkeypatch.setattr(
        "tools.probe_ticker_capabilities.ccxt_async.fakeprobe",
        FakeCCXTExchange,
        raising=False,
    )

    session = create_exchange(
        "fakeprobe",
        {
            "exchange": "fakeprobe",
            "apiKey": "key",
            "secret": "secret",
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
        },
    )

    assert captured["config"]["options"]["defaultType"] == "swap"
    assert captured["config"]["options"]["adjustForTimeDifference"] is True
    assert session.options["defaultType"] == "swap"


@pytest.mark.asyncio
async def test_probe_exchange_ticker_endpoints_records_all_endpoint_shapes():
    class FakeExchange:
        id = "fake"
        rateLimit = 50
        enableRateLimit = True
        has = {
            "fetchBalance": True,
            "fetchBidsAsks": True,
            "fetchMyTrades": True,
            "fetchOHLCV": True,
            "fetchOpenOrders": True,
            "fetchOrderBook": True,
            "fetchPositions": True,
            "fetchTime": True,
            "fetchTicker": True,
            "fetchTickers": True,
        }

        def __init__(self):
            self.fetch_tickers_calls = []

        async def load_markets(self):
            return {
                "BTC/USDT:USDT": {
                    "base": "BTC",
                    "quote": "USDT",
                    "active": True,
                    "swap": True,
                    "linear": True,
                },
                "ETH/USDT:USDT": {
                    "base": "ETH",
                    "quote": "USDT",
                    "active": True,
                    "swap": True,
                    "linear": True,
                },
            }

        async def fetch_time(self):
            return 1_767_225_600_000

        async def fetch_tickers(self, symbols=None):
            self.fetch_tickers_calls.append(symbols)
            selected = symbols or ["BTC/USDT:USDT", "ETH/USDT:USDT"]
            return {symbol: {"symbol": symbol, "bid": 9.0, "ask": 11.0, "last": 10.0} for symbol in selected}

        async def fetch_bids_asks(self, symbols=None):
            selected = symbols or ["BTC/USDT:USDT", "ETH/USDT:USDT"]
            return {symbol: {"symbol": symbol, "bid": 9.0, "ask": 11.0, "last": 10.0} for symbol in selected}

        async def fetch_ticker(self, symbol):
            return {"symbol": symbol, "bid": 9.0, "ask": 11.0, "last": 10.0}

        async def fetch_order_book(self, symbol, limit=None):
            return {"bids": [[9.0, 1.0]], "asks": [[11.0, 1.0]]}

        async def fetch_ohlcv(self, symbol, timeframe="1m", limit=None):
            return [
                [1_000_000, 9.0, 11.0, 8.0, 10.0, 100.0],
                [1_060_000, 10.0, 12.0, 9.0, 11.0, 101.0],
            ]

        async def fetch_balance(self):
            return {"USDT": {"total": 100.0}}

        async def fetch_positions(self):
            return [{"symbol": "BTC/USDT:USDT"}]

        async def fetch_open_orders(self):
            return []

        async def fetch_my_trades(self, symbol=None, since=None, limit=None):
            return [
                {
                    "id": "raw-trade-id-1",
                    "order": "raw-order-id-1",
                    "timestamp": 1_767_225_540_000,
                    "symbol": symbol,
                    "side": "buy",
                },
                {
                    "id": "raw-trade-id-2",
                    "orderId": "raw-order-id-2",
                    "timestamp": 1_767_225_600_000,
                    "symbol": symbol,
                    "side": "sell",
                },
            ]

    exchange = FakeExchange()

    out = await probe_exchange_ticker_endpoints(
        exchange,
        user="fake_user",
        user_info={"quote": "USDT"},
        symbols=[],
        coins=["BTC", "ETH"],
        quote=None,
        max_symbols=5,
        repeats=1,
        sleep_between_seconds=0.0,
    )

    repeat = out["repeats"][0]
    assert out["symbols"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert repeat["fetch_tickers_all"]["ok"] is True
    assert repeat["fetch_tickers_symbols"]["ok"] is True
    assert repeat["fetch_ticker_sequential"]["ok"] is True
    assert repeat["fetch_ticker_concurrent"]["ok"] is True
    assert repeat["fetch_bids_asks_all"]["ok"] is True
    assert repeat["fetch_bids_asks_symbols"]["ok"] is True
    assert repeat["fetch_order_book_sequential"]["ok"] is True
    assert repeat["fetch_order_book_concurrent"]["ok"] is True
    assert repeat["fetch_ohlcv_1m_tail"]["ok"] is True
    assert repeat["fetch_balance"]["ok"] is True
    assert repeat["fetch_positions"]["ok"] is True
    assert repeat["fetch_open_orders_all"]["ok"] is True
    assert repeat["fetch_my_trades_first_symbol"]["ok"] is True
    assert repeat["fetch_my_trades_first_symbol"]["symbol"] == "BTC/USDT:USDT"
    assert repeat["fetch_my_trades_first_symbol"]["value"]["count"] == 2
    assert repeat["fetch_my_trades_first_symbol"]["value"]["id_present_count"] == 2
    assert repeat["fetch_my_trades_first_symbol"]["value"]["order_present_count"] == 2
    assert repeat["fetch_time"]["ok"] is True
    assert repeat["fetch_time"]["supported"] is True
    assert isinstance(repeat["fetch_time"]["clock_skew_ms"], int)
    assert out["time_sync_health"]["enabled"] is True
    assert out["time_sync_health"]["total"] == 1
    assert out["time_sync_health"]["succeeded"] == 1
    assert out["time_sync_health"]["failed"] == 0
    assert out["candle_freshness_health"]["enabled"] is True
    assert out["candle_freshness_health"]["total_symbols"] == 2
    assert out["candle_freshness_health"]["succeeded_symbols"] == 2
    assert out["candle_freshness_health"]["failed_symbols"] == 0
    assert out["candle_freshness_health"]["last_age_ms"]["count"] == 2
    assert out["account_critical_health"]["enabled"] is True
    assert out["account_critical_health"]["total"] == 3
    assert out["account_critical_health"]["succeeded"] == 3
    assert out["account_critical_health"]["failed"] == 0
    assert out["account_critical_health"]["surfaces"]["balance"]["latency_ms"]["count"] == 1
    assert out["fill_history_health"]["enabled"] is True
    assert out["fill_history_health"]["total"] == 1
    assert out["fill_history_health"]["succeeded"] == 1
    assert out["fill_history_health"]["failed"] == 0
    assert out["fill_history_health"]["latest"]["symbol"] == "BTC/USDT:USDT"
    assert out["fill_history_health"]["latest"]["trade_count"] == 2
    assert out["fill_history_health"]["latest"]["id_present_count"] == 2
    assert out["fill_history_health"]["latest"]["order_present_count"] == 2
    assert "raw-trade-id" not in str(out["fill_history_health"])
    assert "raw-order-id" not in str(out["fill_history_health"])
    assert out["rate_limit_health"]["exchange_rate_limit_ms"] == 50.0
    assert out["rate_limit_health"]["exchange_enable_rate_limit"] is True
    assert out["rate_limit_health"]["observed_call_count"] == 20
    assert out["rate_limit_health"]["market_metadata_call_count"] == 1
    assert out["rate_limit_health"]["time_sync_call_count"] == 1
    assert out["rate_limit_health"]["public_call_count"] == 14
    assert out["rate_limit_health"]["private_call_count"] == 4
    assert out["rate_limit_health"]["concurrent_request_count"] == 4
    assert out["rate_limit_health"]["estimated_min_serial_ms"] == 1000.0
    assert out["rate_limit_health"]["endpoint_counts"]["fetch_ticker_concurrent"] == 2
    assert out["rate_limit_health"]["endpoint_counts"]["fetch_open_orders"] == 1
    assert out["rate_limit_health"]["notes"] == [
        "contains_concurrent_batches",
        "contains_authenticated_calls",
    ]
    assert out["endpoint_latency_health"]["total"] == 20
    assert out["endpoint_latency_health"]["failed"] == 0
    assert out["endpoint_latency_health"]["endpoint_count"] == 15
    assert out["endpoint_latency_health"]["endpoints"]["fetch_ticker_concurrent"]["total"] == 2
    assert out["endpoint_latency_health"]["endpoints"]["fetch_open_orders"]["total"] == 1
    assert out["endpoint_latency_health"]["endpoints"]["fetch_my_trades_first_symbol"]["total"] == 1
    assert out["endpoint_latency_health"]["slowest"]["endpoint"] in out["endpoint_latency_health"]["endpoints"]
    assert exchange.fetch_tickers_calls == [None, ["BTC/USDT:USDT", "ETH/USDT:USDT"]]


@pytest.mark.asyncio
async def test_probe_exchange_ticker_endpoints_opt_in_fill_history_pagination():
    class FakeExchange:
        id = "fake"
        rateLimit = 100
        enableRateLimit = True
        has = {
            "fetchBalance": True,
            "fetchMyTrades": True,
            "fetchOpenOrders": True,
            "fetchPositions": True,
        }

        def __init__(self):
            self.my_trades_calls = []

        async def load_markets(self):
            return {
                "BTC/USDT:USDT": {
                    "base": "BTC",
                    "quote": "USDT",
                    "active": True,
                    "swap": True,
                    "linear": True,
                }
            }

        async def fetch_balance(self):
            return {"USDT": {"total": 100.0}}

        async def fetch_positions(self):
            return []

        async def fetch_open_orders(self):
            return []

        async def fetch_my_trades(self, symbol=None, since=None, limit=None):
            self.my_trades_calls.append((symbol, since, limit))
            if since is None:
                return [
                    {
                        "id": "raw-trade-id-1",
                        "order": "raw-order-id-1",
                        "timestamp": 1_000,
                        "symbol": symbol,
                        "side": "buy",
                    },
                    {
                        "id": "raw-trade-id-2",
                        "order": "raw-order-id-2",
                        "timestamp": 2_000,
                        "symbol": symbol,
                        "side": "sell",
                    },
                ]
            return [
                {
                    "id": "raw-trade-id-3",
                    "order": "raw-order-id-3",
                    "timestamp": 3_000,
                    "symbol": symbol,
                    "side": "buy",
                }
            ]

    exchange = FakeExchange()

    out = await probe_exchange_ticker_endpoints(
        exchange,
        user="fake_user",
        user_info={"quote": "USDT"},
        symbols=[],
        coins=[],
        quote=None,
        max_symbols=1,
        repeats=1,
        sleep_between_seconds=0.0,
        include_public=False,
        include_order_book=False,
        include_ohlcv=False,
        include_time_sync=False,
        fill_history_pages=3,
        fill_history_page_limit=2,
    )

    outcome = out["repeats"][0]["fetch_my_trades_first_symbol"]
    assert exchange.my_trades_calls == [
        ("BTC/USDT:USDT", None, 2),
        ("BTC/USDT:USDT", 2_001, 2),
    ]
    assert outcome["ok"] is True
    assert outcome["call_count"] == 2
    assert outcome["requested_pages"] == 3
    assert outcome["page_limit"] == 2
    assert outcome["page_count"] == 2
    assert outcome["terminal_reason"] == "short_page"
    assert outcome["value"]["count"] == 3
    assert outcome["value"]["id_present_count"] == 3
    assert outcome["value"]["order_present_count"] == 3
    assert out["fill_history_health"]["latest"]["call_count"] == 2
    assert out["fill_history_health"]["latest"]["page_count"] == 2
    assert out["fill_history_health"]["latest"]["terminal_reason"] == "short_page"
    assert out["fill_history_health"]["latest"]["trade_count"] == 3
    assert out["rate_limit_health"]["observed_call_count"] == 6
    assert out["rate_limit_health"]["private_call_count"] == 5
    assert out["rate_limit_health"]["endpoint_counts"]["fetch_my_trades_first_symbol"] == 2
    assert out["endpoint_latency_health"]["total"] == 6
    assert out["endpoint_latency_health"]["endpoints"]["fetch_my_trades_first_symbol"]["total"] == 2
    assert out["endpoint_latency_health"]["endpoints"]["fetch_my_trades_first_symbol"]["failed"] == 0
    assert "raw-trade-id" not in str(out)
    assert "raw-order-id" not in str(out)
    assert "raw-trade-id" not in str(out["fill_history_health"])
    assert "raw-order-id" not in str(out["fill_history_health"])


def test_account_critical_probe_health_summarizes_failures_without_raw_errors():
    summary = summarize_account_critical_probe_health(
        {
            "include_private": True,
            "repeats": [
                {
                    "fetch_balance": {
                        "ok": True,
                        "elapsed_ms": 1.25,
                        "value": {"redacted": True},
                    },
                    "fetch_positions": {
                        "ok": False,
                        "elapsed_ms": 2.5,
                        "error_type": "RequestTimeout",
                        "error": "https://api.exchange.invalid/path?secret=leak",
                    },
                    "fetch_open_orders_all": {
                        "ok": False,
                        "elapsed_ms": 3.75,
                        "error_type": "RuntimeError",
                        "error": "raw payload should not appear",
                    },
                }
            ],
        }
    )

    assert summary["enabled"] is True
    assert summary["total"] == 3
    assert summary["succeeded"] == 1
    assert summary["failed"] == 2
    assert summary["failure_pct"] == pytest.approx(66.667)
    assert summary["surfaces"]["positions"]["error_types"] == {"RequestTimeout": 1}
    assert summary["surfaces"]["open_orders"]["latest"]["error_type"] == "RuntimeError"
    assert "secret=leak" not in str(summary)
    assert "raw payload" not in str(summary)


def test_endpoint_latency_probe_health_summarizes_existing_outcomes_only():
    summary = summarize_endpoint_latency_probe_health(
        {
            "load_markets": {"ok": True, "elapsed_ms": 10.0},
            "repeats": [
                {
                    "fetch_tickers_all": {"ok": True, "elapsed_ms": 20.0},
                    "fetch_ticker_concurrent": {
                        "symbols": {
                            "BTC/USDT:USDT": {"ok": True, "elapsed_ms": 30.0},
                            "ETH/USDT:USDT": {
                                "ok": False,
                                "elapsed_ms": 40.0,
                                "error_type": "RequestTimeout",
                                "error": "raw exchange error should not appear",
                            },
                        }
                    },
                    "fetch_open_orders_all": {
                        "attempts": {
                            "all_symbols": {
                                "ok": False,
                                "elapsed_ms": 0.5,
                                "error_type": "ExchangeError",
                                "error": "raw warning should not appear",
                            },
                            "symbol": {
                                "symbol": "BTC/USDT:USDT",
                                "outcome": {"ok": True, "elapsed_ms": 12.5},
                            },
                        }
                    },
                    "fetch_my_trades_first_symbol": {
                        "pages": [
                            {"ok": True, "elapsed_ms": 5.0, "trade_count": 2},
                            {"ok": True, "elapsed_ms": 6.0, "trade_count": 1},
                        ]
                    },
                }
            ],
        }
    )

    assert summary["total"] == 8
    assert summary["succeeded"] == 6
    assert summary["failed"] == 2
    assert summary["failure_pct"] == pytest.approx(25.0)
    assert summary["endpoints"]["fetch_ticker_concurrent"]["total"] == 2
    assert summary["endpoints"]["fetch_ticker_concurrent"]["failed"] == 1
    assert summary["endpoints"]["fetch_ticker_concurrent"]["error_types"] == {
        "RequestTimeout": 1
    }
    assert summary["endpoints"]["fetch_open_orders"]["total"] == 2
    assert summary["endpoints"]["fetch_my_trades_first_symbol"]["total"] == 2
    assert summary["slowest"]["endpoint"] == "fetch_ticker_concurrent"
    assert "raw exchange error" not in str(summary)
    assert "raw warning" not in str(summary)


def test_endpoint_latency_probe_collection_aggregates_raw_probe_latencies():
    collection = summarize_endpoint_latency_probe_collection(
        [
            {
                "user": "binance_01",
                "exchange": "binance",
                "load_markets": {"ok": True, "elapsed_ms": 10.0},
                "repeats": [{"fetch_balance": {"ok": True, "elapsed_ms": 20.0}}],
            },
            {
                "user": "kucoin_01",
                "exchange": "kucoinfutures",
                "load_markets": {"ok": True, "elapsed_ms": 30.0},
                "repeats": [
                    {
                        "fetch_balance": {
                            "ok": False,
                            "elapsed_ms": 40.0,
                            "error_type": "RequestTimeout",
                        }
                    }
                ],
            },
        ]
    )

    assert collection["total"] == 4
    assert collection["succeeded"] == 3
    assert collection["failed"] == 1
    assert collection["users"][0]["user"] == "binance_01"
    assert collection["endpoints"]["load_markets"]["total"] == 2
    assert collection["endpoints"]["fetch_balance"]["total"] == 2
    assert collection["endpoints"]["fetch_balance"]["failed"] == 1
    assert collection["endpoints"]["fetch_balance"]["error_types"] == {
        "RequestTimeout": 1
    }


def test_account_critical_probe_collection_aggregates_user_summaries():
    probes = [
        {
            "user": "kucoin_01",
            "exchange": "kucoinfutures",
            "include_private": True,
            "repeats": [
                {
                    "fetch_balance": {"ok": True, "elapsed_ms": 10.0},
                    "fetch_positions": {"ok": True, "elapsed_ms": 20.0},
                    "fetch_open_orders_all": {"ok": False, "error_type": "RequestTimeout"},
                }
            ],
        },
        {
            "user": "binance_01",
            "exchange": "binance",
            "include_private": False,
            "repeats": [{"fetch_tickers_all": {"ok": True}}],
        },
    ]

    collection = summarize_account_critical_probe_collection(probes)

    assert collection["total"] == 3
    assert collection["succeeded"] == 2
    assert collection["failed"] == 1
    assert collection["users"][0]["user"] == "kucoin_01"
    assert collection["users"][0]["exchange"] == "kucoinfutures"
    assert collection["users"][0]["enabled"] is True
    assert collection["users"][0]["total"] == 3
    assert collection["users"][0]["succeeded"] == 2
    assert collection["users"][0]["failed"] == 1
    assert collection["users"][0]["failure_pct"] == pytest.approx(33.333)
    assert collection["users"][1] == {
        "user": "binance_01",
        "exchange": "binance",
        "enabled": False,
        "total": 0,
        "succeeded": 0,
        "failed": 0,
        "failure_pct": None,
    }


def test_time_sync_probe_health_summarizes_success_failure_and_unsupported():
    summary = summarize_time_sync_probe_health(
        {
            "include_time_sync": True,
            "repeats": [
                {
                    "fetch_time": {
                        "ok": True,
                        "supported": True,
                        "elapsed_ms": 2.5,
                        "clock_skew_ms": -125,
                    }
                },
                {
                    "fetch_time": {
                        "ok": False,
                        "supported": True,
                        "elapsed_ms": 3.75,
                        "error_type": "RequestTimeout",
                        "error": "raw exchange error should not be summarized",
                    }
                },
                {
                    "fetch_time": {
                        "ok": False,
                        "supported": False,
                        "skipped": True,
                        "error_type": "UnsupportedEndpoint",
                    }
                },
            ],
        }
    )

    assert summary["enabled"] is True
    assert summary["total"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["unsupported"] == 1
    assert summary["failure_pct"] == pytest.approx(50.0)
    assert summary["clock_skew_ms"]["max_abs"] == 125.0
    assert summary["error_types"] == {"RequestTimeout": 1}
    assert "raw exchange error" not in str(summary)


def test_time_sync_probe_collection_aggregates_user_summaries():
    collection = summarize_time_sync_probe_collection(
        [
            {
                "user": "okx_faisal",
                "exchange": "okx",
                "time_sync_health": {
                    "enabled": True,
                    "total": 1,
                    "succeeded": 1,
                    "failed": 0,
                    "unsupported": 0,
                    "failure_pct": 0.0,
                    "clock_skew_ms": {"max_abs": 42.0},
                    "latest": {"ok": True, "clock_skew_ms": 42.0},
                },
            },
            {
                "user": "legacy_01",
                "exchange": "legacy",
                "time_sync_health": {
                    "enabled": True,
                    "total": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "unsupported": 1,
                    "failure_pct": None,
                    "clock_skew_ms": {"max_abs": None},
                    "latest": {"ok": False, "supported": False, "skipped": True},
                },
            },
        ]
    )

    assert collection["total"] == 1
    assert collection["succeeded"] == 1
    assert collection["failed"] == 0
    assert collection["unsupported"] == 1
    assert collection["clock_skew_ms"]["max_abs"] == 42.0
    assert collection["users"][0]["latest_clock_skew_ms"] == 42.0
    assert collection["users"][1]["unsupported"] == 1


def test_candle_freshness_probe_health_summarizes_symbol_tail_ages():
    summary = summarize_candle_freshness_probe_health(
        {
            "include_public": True,
            "include_ohlcv": True,
            "repeats": [
                {
                    "fetch_ohlcv_1m_tail": {
                        "ok": False,
                        "elapsed_ms": 12.0,
                        "symbols": {
                            "BTC/USDT:USDT": {
                                "ok": True,
                                "elapsed_ms": 4.5,
                                "value": {
                                    "last_timestamp": 1_000_000,
                                    "last_datetime": "1970-01-01T00:16:40",
                                    "last_age_ms": 60_000,
                                    "last_is_current_incomplete_minute": False,
                                },
                            },
                            "ETH/USDT:USDT": {
                                "ok": True,
                                "elapsed_ms": 5.5,
                                "value": {
                                    "last_timestamp": 1_060_000,
                                    "last_datetime": "1970-01-01T00:17:40",
                                    "last_age_ms": 10_000,
                                    "last_is_current_incomplete_minute": True,
                                },
                            },
                            "XRP/USDT:USDT": {
                                "ok": False,
                                "elapsed_ms": 6.5,
                                "error_type": "RequestTimeout",
                                "error": "raw exchange error should not appear",
                            },
                        },
                    }
                }
            ],
        }
    )

    assert summary["enabled"] is True
    assert summary["total_symbols"] == 3
    assert summary["succeeded_symbols"] == 2
    assert summary["failed_symbols"] == 1
    assert summary["failure_pct"] == pytest.approx(33.333)
    assert summary["current_incomplete_symbols"] == 1
    assert summary["missing_timestamp_symbols"] == 0
    assert summary["last_age_ms"]["max"] == 60_000.0
    assert summary["worst"]["symbol"] == "BTC/USDT:USDT"
    assert summary["error_types"] == {"RequestTimeout": 1}
    assert "raw exchange error" not in str(summary)


def test_candle_freshness_probe_collection_aggregates_users():
    probes = [
        {
            "user": "binance_01",
            "exchange": "binance",
            "include_public": True,
            "include_ohlcv": True,
            "repeats": [
                {
                    "fetch_ohlcv_1m_tail": {
                        "symbols": {
                            "BTC/USDT:USDT": {
                                "ok": True,
                                "value": {
                                    "last_age_ms": 30_000,
                                    "last_datetime": "fresh",
                                    "last_is_current_incomplete_minute": True,
                                },
                            }
                        }
                    }
                }
            ],
        },
        {
            "user": "okx_faisal",
            "exchange": "okx",
            "include_public": True,
            "include_ohlcv": True,
            "repeats": [
                {
                    "fetch_ohlcv_1m_tail": {
                        "symbols": {
                            "ETH/USDT:USDT": {
                                "ok": True,
                                "value": {
                                    "last_age_ms": 120_000,
                                    "last_datetime": "stale",
                                    "last_is_current_incomplete_minute": False,
                                },
                            }
                        }
                    }
                }
            ],
        },
    ]

    collection = summarize_candle_freshness_probe_collection(probes)

    assert collection["total_symbols"] == 2
    assert collection["succeeded_symbols"] == 2
    assert collection["failed_symbols"] == 0
    assert collection["current_incomplete_symbols"] == 1
    assert collection["last_age_ms"]["max"] == 120_000.0
    assert collection["worst"] == {
        "user": "okx_faisal",
        "exchange": "okx",
        "symbol": "ETH/USDT:USDT",
        "last_age_ms": 120_000.0,
        "last_datetime": "stale",
        "last_is_current_incomplete_minute": False,
    }
    assert collection["users"][0]["max_last_age_ms"] == 30_000.0
    assert collection["users"][1]["worst_symbol"] == "ETH/USDT:USDT"


def test_summarize_my_trades_reports_shape_without_raw_ids():
    summary = summarize_my_trades(
        [
            {
                "id": "raw-fill-id-a",
                "order": "raw-order-id-a",
                "timestamp": 1_000_000,
                "symbol": "BTC/USDT:USDT",
                "side": "buy",
                "info": {"secret": "not emitted"},
            },
            {
                "id": "raw-fill-id-b",
                "orderId": "raw-order-id-b",
                "timestamp": 1_060_000,
                "symbol": "ETH/USDT:USDT",
                "side": "sell",
            },
            {"symbol": "ETH/USDT:USDT"},
        ]
    )

    assert summary["count"] == 3
    assert summary["dict_count"] == 3
    assert summary["timestamp_count"] == 2
    assert summary["missing_timestamp_count"] == 1
    assert summary["first_timestamp"] == 1_000_000
    assert summary["last_timestamp"] == 1_060_000
    assert summary["symbol_count"] == 2
    assert summary["symbols_sample"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert summary["side_counts"] == {"buy": 1, "sell": 1}
    assert summary["id_present_count"] == 2
    assert summary["order_present_count"] == 2
    assert "raw-fill-id" not in str(summary)
    assert "raw-order-id" not in str(summary)
    assert "not emitted" not in str(summary)


def test_fill_history_probe_health_summarizes_success_failure_without_raw_errors():
    summary = summarize_fill_history_probe_health(
        {
            "include_private": True,
            "include_my_trades": True,
            "repeats": [
                {
                    "fetch_my_trades_first_symbol": {
                        "ok": True,
                        "symbol": "BTC/USDT:USDT",
                        "elapsed_ms": 12.0,
                        "value": {
                            "count": 2,
                            "timestamp_count": 2,
                            "missing_timestamp_count": 0,
                            "first_timestamp": 1_000_000,
                            "first_datetime": "1970-01-01T00:16:40",
                            "last_timestamp": 1_060_000,
                            "last_datetime": "1970-01-01T00:17:40",
                            "symbol_count": 1,
                            "side_counts": {"buy": 1, "sell": 1},
                            "id_present_count": 2,
                            "order_present_count": 2,
                        },
                    }
                },
                {
                    "fetch_my_trades_first_symbol": {
                        "ok": False,
                        "symbol": "ETH/USDT:USDT",
                        "elapsed_ms": 22.0,
                        "error_type": "RequestTimeout",
                        "error": "raw exchange error apiKey=SECRET should not appear",
                    }
                },
            ],
        }
    )

    assert summary["enabled"] is True
    assert summary["total"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["failure_pct"] == pytest.approx(50.0)
    assert summary["latency_ms"]["count"] == 2
    assert summary["trade_count"]["max"] == 2.0
    assert summary["error_types"] == {"RequestTimeout": 1}
    assert summary["latest"]["ok"] is False
    assert summary["latest"]["error_type"] == "RequestTimeout"
    assert summary["newest_trade"]["symbol"] == "BTC/USDT:USDT"
    assert summary["newest_trade"]["last_timestamp"] == 1_060_000
    assert "raw exchange error" not in str(summary)
    assert "SECRET" not in str(summary)


def test_fill_history_probe_collection_aggregates_users():
    collection = summarize_fill_history_probe_collection(
        [
            {
                "user": "binance_01",
                "exchange": "binance",
                "fill_history_health": {
                    "enabled": True,
                    "total": 1,
                    "succeeded": 1,
                    "failed": 0,
                    "failure_pct": 0.0,
                    "latest": {
                        "symbol": "BTC/USDT:USDT",
                        "trade_count": 2,
                    },
                    "newest_trade": {
                        "symbol": "BTC/USDT:USDT",
                        "last_timestamp": 1_060_000,
                        "last_datetime": "1970-01-01T00:17:40",
                        "trade_count": 2,
                    },
                },
            },
            {
                "user": "okx_01",
                "exchange": "okx",
                "fill_history_health": {
                    "enabled": True,
                    "total": 1,
                    "succeeded": 0,
                    "failed": 1,
                    "failure_pct": 100.0,
                    "latest": {
                        "symbol": "ETH/USDT:USDT",
                        "trade_count": 0,
                    },
                    "newest_trade": None,
                },
            },
            {
                "user": "paper_01",
                "exchange": "paper",
                "fill_history_health": {
                    "enabled": False,
                    "total": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "failure_pct": None,
                    "latest": None,
                    "newest_trade": None,
                },
            },
        ]
    )

    assert collection["total"] == 2
    assert collection["succeeded"] == 1
    assert collection["failed"] == 1
    assert collection["failure_pct"] == pytest.approx(50.0)
    assert collection["trade_count"]["count"] == 2
    assert collection["newest_trade"] == {
        "user": "binance_01",
        "exchange": "binance",
        "symbol": "BTC/USDT:USDT",
        "last_timestamp": 1_060_000,
        "last_datetime": "1970-01-01T00:17:40",
        "trade_count": 2,
    }
    assert collection["users"][2]["enabled"] is False


def test_rate_limit_probe_health_estimates_existing_probe_pressure():
    summary = summarize_rate_limit_probe_health(
        {
            "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            "sleep_between_seconds": 1.5,
            "exchange_rate_limit_ms": 25,
            "exchange_enable_rate_limit": False,
            "load_markets": {"ok": True},
            "repeats": [
                {
                    "fetch_time": {"ok": True},
                    "fetch_tickers_all": {"ok": True},
                    "fetch_tickers_symbols": {"ok": True},
                    "fetch_ticker_sequential": {
                        "symbols": {
                            "BTC/USDT:USDT": {"ok": True},
                            "ETH/USDT:USDT": {"ok": True},
                        }
                    },
                    "fetch_ticker_concurrent": {
                        "symbols": {
                            "BTC/USDT:USDT": {"ok": True},
                            "ETH/USDT:USDT": {"ok": True},
                        }
                    },
                    "fetch_bids_asks_all": {"ok": True},
                    "fetch_bids_asks_symbols": {"ok": True},
                    "fetch_order_book_sequential": {
                        "symbols": {
                            "BTC/USDT:USDT": {"ok": True},
                            "ETH/USDT:USDT": {"ok": True},
                        }
                    },
                    "fetch_order_book_concurrent": {
                        "symbols": {
                            "BTC/USDT:USDT": {"ok": True},
                            "ETH/USDT:USDT": {"ok": True},
                        }
                    },
                    "fetch_ohlcv_1m_tail": {
                        "symbols": {
                            "BTC/USDT:USDT": {"ok": True},
                            "ETH/USDT:USDT": {"ok": True},
                        }
                    },
                    "fetch_balance": {"ok": True},
                    "fetch_positions": {"ok": True},
                    "fetch_open_orders_all": {
                        "ok": True,
                        "attempts": {"all_symbols": {"ok": True}},
                    },
                    "fetch_my_trades_first_symbol": {"ok": True},
                }
            ],
        }
    )

    assert summary["observed_call_count"] == 20
    assert summary["market_metadata_call_count"] == 1
    assert summary["time_sync_call_count"] == 1
    assert summary["public_call_count"] == 14
    assert summary["private_call_count"] == 4
    assert summary["concurrent_request_count"] == 4
    assert summary["calls_per_repeat"]["max"] == 19.0
    assert summary["estimated_min_serial_ms"] == 500.0
    assert summary["configured_sleep_between_seconds"] == 1.5
    assert summary["endpoint_counts"]["fetch_open_orders"] == 1
    assert summary["concurrent_endpoint_counts"] == {
        "fetch_order_book_concurrent": 2,
        "fetch_ticker_concurrent": 2,
    }
    assert summary["notes"] == [
        "ccxt_rate_limit_disabled",
        "contains_concurrent_batches",
        "contains_authenticated_calls",
    ]


def test_rate_limit_probe_collection_aggregates_users():
    collection = summarize_rate_limit_probe_collection(
        [
            {
                "user": "binance_01",
                "exchange": "binance",
                "rate_limit_health": {
                    "exchange_rate_limit_ms": 50.0,
                    "exchange_enable_rate_limit": True,
                    "observed_call_count": 20,
                    "private_call_count": 4,
                    "public_call_count": 14,
                    "concurrent_request_count": 4,
                    "estimated_min_serial_ms": 1000.0,
                    "endpoint_counts": {"fetch_ticker_concurrent": 2},
                    "concurrent_endpoint_counts": {"fetch_ticker_concurrent": 2},
                    "notes": ["contains_concurrent_batches"],
                },
            },
            {
                "user": "gateio_01",
                "exchange": "gateio",
                "rate_limit_health": {
                    "exchange_rate_limit_ms": 100.0,
                    "exchange_enable_rate_limit": True,
                    "observed_call_count": 5,
                    "private_call_count": 4,
                    "public_call_count": 0,
                    "concurrent_request_count": 0,
                    "estimated_min_serial_ms": 500.0,
                    "endpoint_counts": {"fetch_open_orders": 2},
                    "concurrent_endpoint_counts": {},
                    "notes": ["contains_authenticated_calls"],
                },
            },
        ]
    )

    assert collection["observed_call_count"] == 25
    assert collection["public_call_count"] == 14
    assert collection["private_call_count"] == 8
    assert collection["concurrent_request_count"] == 4
    assert collection["exchange_rate_limit_ms"]["max"] == 100.0
    assert collection["estimated_min_serial_ms"]["median"] == 750.0
    assert collection["endpoint_counts"] == {
        "fetch_open_orders": 2,
        "fetch_ticker_concurrent": 2,
    }
    assert collection["concurrent_endpoint_counts"] == {"fetch_ticker_concurrent": 2}
    assert collection["users"][0]["notes"] == ["contains_concurrent_batches"]


@pytest.mark.asyncio
async def test_time_sync_probe_uses_ccxt_capability_flag_for_unsupported_method():
    class FakeExchange:
        id = "fake"
        rateLimit = 100
        enableRateLimit = True
        has = {
            "fetchBalance": True,
            "fetchBidsAsks": True,
            "fetchMyTrades": True,
            "fetchOpenOrders": True,
            "fetchPositions": True,
            "fetchTime": False,
            "fetchTicker": True,
            "fetchTickers": True,
        }

        def __init__(self):
            self.fetch_time_called = False

        async def load_markets(self):
            return {
                "BTC/USDT:USDT": {
                    "base": "BTC",
                    "quote": "USDT",
                    "active": True,
                    "swap": True,
                    "linear": True,
                }
            }

        async def fetch_time(self):
            self.fetch_time_called = True
            raise RuntimeError("should not be called when has.fetchTime is false")

        async def fetch_balance(self):
            return {"USDT": {"total": 100.0}}

        async def fetch_positions(self):
            return []

        async def fetch_open_orders(self, symbol=None):
            return []

    exchange = FakeExchange()

    out = await probe_exchange_ticker_endpoints(
        exchange,
        user="fake_user",
        user_info={"quote": "USDT"},
        symbols=["BTC/USDT:USDT"],
        coins=[],
        quote=None,
        max_symbols=1,
        repeats=1,
        sleep_between_seconds=0.0,
        include_public=False,
        include_order_book=False,
        include_ohlcv=False,
        include_my_trades=False,
    )

    assert exchange.fetch_time_called is False
    assert out["repeats"][0]["fetch_time"]["supported"] is False
    assert out["repeats"][0]["fetch_time"]["skipped"] is True
    assert out["time_sync_health"]["total"] == 0
    assert out["time_sync_health"]["failed"] == 0
    assert out["time_sync_health"]["unsupported"] == 1


@pytest.mark.asyncio
async def test_probe_exchange_ticker_endpoints_redacts_stored_endpoint_errors():
    class FakeExchange:
        id = "fake"
        has = {
            "fetchBalance": True,
            "fetchBidsAsks": True,
            "fetchMyTrades": True,
            "fetchOpenOrders": True,
            "fetchPositions": True,
            "fetchTicker": True,
            "fetchTickers": True,
        }

        async def load_markets(self):
            return {
                "BTC/USDT:USDT": {
                    "base": "BTC",
                    "quote": "USDT",
                    "active": True,
                    "swap": True,
                    "linear": True,
                }
            }

        async def fetch_tickers(self, symbols=None):
            selected = symbols or ["BTC/USDT:USDT"]
            return {symbol: {"symbol": symbol, "bid": 9.0, "ask": 11.0, "last": 10.0} for symbol in selected}

        async def fetch_bids_asks(self, symbols=None):
            selected = symbols or ["BTC/USDT:USDT"]
            return {symbol: {"symbol": symbol, "bid": 9.0, "ask": 11.0, "last": 10.0} for symbol in selected}

        async def fetch_ticker(self, symbol):
            return {"symbol": symbol, "bid": 9.0, "ask": 11.0, "last": 10.0}

        async def fetch_balance(self):
            raise RuntimeError(
                "GET https://api.exchange.invalid/account?apiKey=SECRET&signature=SIG "
                "Authorization: Bearer TOKEN"
            )

        async def fetch_positions(self):
            return []

        async def fetch_open_orders(self):
            return []

        async def fetch_my_trades(self, symbol=None, since=None, limit=None):
            return []

    out = await probe_exchange_ticker_endpoints(
        FakeExchange(),
        user="fake_user",
        user_info={"quote": "USDT"},
        symbols=["BTC/USDT:USDT"],
        coins=[],
        quote=None,
        max_symbols=1,
        repeats=1,
        sleep_between_seconds=0.0,
        include_order_book=False,
        include_ohlcv=False,
    )

    error = out["repeats"][0]["fetch_balance"]["error"]
    assert "SECRET" not in error
    assert "SIG" not in error
    assert "TOKEN" not in error
    assert "apiKey=[redacted]" in error
    assert "signature=[redacted]" in error
    assert "Authorization: [redacted]" in error
    assert out["account_critical_health"]["surfaces"]["balance"]["error_types"] == {
        "RuntimeError": 1
    }


@pytest.mark.asyncio
async def test_account_only_probe_skips_public_and_uses_open_orders_symbol_fallback():
    class FakeExchange:
        id = "fake"
        has = {
            "fetchBalance": True,
            "fetchBidsAsks": True,
            "fetchMyTrades": True,
            "fetchOpenOrders": True,
            "fetchPositions": True,
            "fetchTicker": True,
            "fetchTickers": True,
        }

        def __init__(self):
            self.open_order_symbols = []
            self.my_trades_called = False

        async def load_markets(self):
            return {
                "BTC/USDT:USDT": {
                    "base": "BTC",
                    "quote": "USDT",
                    "active": True,
                    "swap": True,
                    "linear": True,
                }
            }

        async def fetch_tickers(self, symbols=None):
            raise AssertionError("account-only probe must not fetch tickers")

        async def fetch_bids_asks(self, symbols=None):
            raise AssertionError("account-only probe must not fetch bids/asks")

        async def fetch_ticker(self, symbol):
            raise AssertionError("account-only probe must not fetch ticker")

        async def fetch_order_book(self, symbol, limit=None):
            raise AssertionError("account-only probe must not fetch order book")

        async def fetch_ohlcv(self, symbol, timeframe="1m", limit=None):
            raise AssertionError("account-only probe must not fetch OHLCV")

        async def fetch_balance(self):
            return {"USDT": {"total": 100.0}}

        async def fetch_positions(self):
            return [{"symbol": "BTC/USDT:USDT"}]

        async def fetch_open_orders(self, symbol=None):
            self.open_order_symbols.append(symbol)
            if symbol is None:
                raise RuntimeError("symbol required apiKey=SECRET")
            return []

        async def fetch_my_trades(self, symbol=None, since=None, limit=None):
            self.my_trades_called = True
            raise AssertionError("account-only probe must not fetch my trades")

    exchange = FakeExchange()

    out = await probe_exchange_ticker_endpoints(
        exchange,
        user="fake_user",
        user_info={"quote": "USDT"},
        symbols=[],
        coins=[],
        quote=None,
        max_symbols=1,
        repeats=1,
        sleep_between_seconds=0.0,
        include_public=False,
        include_order_book=False,
        include_ohlcv=False,
        include_my_trades=False,
    )

    repeat = out["repeats"][0]
    assert out["symbols"] == ["BTC/USDT:USDT"]
    assert out["include_public"] is False
    assert out["include_my_trades"] is False
    assert "fetch_tickers_all" not in repeat
    assert "fetch_my_trades_first_symbol" not in repeat
    assert repeat["fetch_open_orders_all"]["ok"] is True
    assert repeat["fetch_open_orders_all"]["mode"] == "symbol_fallback"
    assert repeat["fetch_open_orders_all"]["fallback_symbol"] == "BTC/USDT:USDT"
    assert "SECRET" not in repeat["fetch_open_orders_all"]["attempts"]["all_symbols"]["error"]
    assert exchange.open_order_symbols == [None, "BTC/USDT:USDT"]
    assert exchange.my_trades_called is False
    assert out["account_critical_health"]["succeeded"] == 3
    assert out["account_critical_health"]["failed"] == 0
    assert out["candle_freshness_health"]["enabled"] is False
    assert out["candle_freshness_health"]["total_symbols"] == 0
    assert out["fill_history_health"]["enabled"] is False
    assert out["fill_history_health"]["total"] == 0
    assert out["rate_limit_health"]["observed_call_count"] == 5
    assert out["rate_limit_health"]["private_call_count"] == 4
    assert out["rate_limit_health"]["public_call_count"] == 0
    assert out["rate_limit_health"]["endpoint_counts"]["fetch_open_orders"] == 2
