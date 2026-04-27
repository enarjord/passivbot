import pytest

from tools.probe_ticker_capabilities import (
    probe_ticker_capabilities,
    summarize_order_book,
    summarize_ticker,
    summarize_tickers,
)
from tools.probe_ccxt_ticker_endpoints import (
    probe_exchange_ticker_endpoints,
    select_default_symbols,
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
        "ETH/USDT:USDT": {"quote": "USDT", "active": True, "contract": True, "linear": True},
        "BTC/USD:BTC": {"quote": "USD", "active": True, "swap": True, "linear": False},
        "OLD/USDT:USDT": {"quote": "USDT", "active": False, "swap": True, "linear": True},
        "SPOT/USDT": {"quote": "USDT", "active": True, "spot": True},
    }

    assert select_default_symbols(markets, quote="USDT", max_symbols=2) == [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
    ]


@pytest.mark.asyncio
async def test_probe_exchange_ticker_endpoints_records_all_endpoint_shapes():
    class FakeExchange:
        id = "fake"
        has = {
            "fetchBalance": True,
            "fetchBidsAsks": True,
            "fetchMyTrades": True,
            "fetchOHLCV": True,
            "fetchOpenOrders": True,
            "fetchOrderBook": True,
            "fetchPositions": True,
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
            return []

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
    assert exchange.fetch_tickers_calls == [None, ["BTC/USDT:USDT", "ETH/USDT:USDT"]]
