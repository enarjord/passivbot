from __future__ import annotations

import json
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ccxt
import pytest

from ccxt_contracts import build_contract_bot, get_bot_class
from exchanges.weex import AsyncWeex, ProWeex, WeexBot
from fill_events_manager import WeexFetcher, _build_fetcher_for_bot
from market_snapshot import MarketSnapshotProvider
from passivbot import custom_id_has_explicit_passivbot_marker, custom_id_to_snake


SYMBOL = "BTC/USDT:USDT"


def _market() -> dict:
    return {
        "id": "BTCUSDT",
        "symbol": SYMBOL,
        "base": "BTC",
        "quote": "USDT",
        "settle": "USDT",
        "type": "swap",
        "spot": False,
        "swap": True,
        "future": False,
        "contract": True,
        "linear": True,
        "inverse": False,
        "active": True,
        "precision": {"amount": 0.0001, "price": 0.1},
        "limits": {
            "amount": {"min": 0.0001, "max": 1200.0},
            "price": {"min": None, "max": None},
            "cost": {"min": None, "max": None},
            "leverage": {"min": 1.0, "max": 400.0},
        },
        "contractSize": 0.0001,
        "info": {"maxLeverage": "400", "contractVal": "0.0001"},
    }


def _ccxt_exchange() -> ccxt.weex:
    exchange = ccxt.weex(
        {"apiKey": "key", "secret": "secret", "password": "passphrase"}
    )
    market = _market()
    exchange.markets = {SYMBOL: market}
    exchange.markets_by_id = {"BTCUSDT": [market]}
    exchange.symbols = [SYMBOL]
    return exchange


def _bot(*, time_in_force: str = "gtc") -> WeexBot:
    bot = build_contract_bot("weex")
    bot.config["live"]["time_in_force"] = time_in_force
    bot.config["live"]["exchange_symbol_unavailable_cooldown_hours"] = 6.0
    bot.broker_code = "WEEX111164"
    bot.custom_id_max_length = 36
    return bot


def test_ccxt_contract_registry_uses_weex_adapter():
    assert get_bot_class("weex") is WeexBot


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class", [AsyncWeex, ProWeex])
async def test_weex_clients_use_ipv4_transport(client_class):
    exchange = client_class()
    try:
        exchange.open()
        assert exchange.tcp_connector._family == socket.AF_INET
    finally:
        await exchange.close()


def test_weex_client_accepts_only_documented_success_envelope():
    exchange = AsyncWeex()
    assert (
        exchange.handle_errors(
            200,
            "OK",
            "https://api-contract.weex.com/capi/v3/account/leverage",
            "POST",
            {},
            '{"code":"200","msg":"success"}',
            {"code": "200", "msg": "success"},
            {},
            "{}",
        )
        is None
    )

    with pytest.raises(ccxt.ExchangeError):
        exchange.handle_errors(
            400,
            "Bad Request",
            "https://api-contract.weex.com/capi/v3/account/leverage",
            "POST",
            {},
            '{"code":"-3313","msg":"bad leverage"}',
            {"code": "-3313", "msg": "bad leverage"},
            {},
            "{}",
        )


def test_weex_classifies_only_exact_api_symbol_unavailable_code():
    bot = _bot()
    unavailable = ccxt.PermissionDenied(
        'weex {"code":-1058,"msg":"The trading pair is not supported via the API"}'
    )
    other = ccxt.PermissionDenied(
        'weex {"code":-1056,"msg":"illegal IP"}'
    )

    assert (
        bot._classify_exchange_symbol_unavailable_error(unavailable)
        == "weex_api_symbol_unavailable"
    )
    assert bot._classify_exchange_symbol_unavailable_error(other) is None
    assert (
        bot._classify_exchange_symbol_unavailable_error(
            ccxt.PermissionDenied("contains -1058 but no structured code")
        )
        is None
    )


def test_weex_symbol_unavailable_cooldown_is_ram_only_scoped_and_expires(caplog):
    bot = _bot()
    bot._exchange_config_retry_after_ms = {}
    bot._exchange_config_retry_attempts = {SYMBOL: 3}
    bot.has_position = lambda symbol=None: False
    error = ccxt.PermissionDenied(
        'weex {"code":-1058,"msg":"The trading pair is not supported via the API"}'
    )
    now_ms = 1_000_000

    assert bot._activate_exchange_symbol_unavailable_cooldown(
        SYMBOL, error, now_ms=now_ms
    )
    expected_until = now_ms + 6 * 60 * 60 * 1000
    assert bot._exchange_symbol_unavailable_until_ms == {SYMBOL: expected_until}
    assert bot._exchange_config_retry_after_ms == {SYMBOL: expected_until}
    assert bot._activate_exchange_symbol_unavailable_cooldown(
        SYMBOL, error, now_ms=now_ms + 60_000
    )
    refreshed_until = expected_until + 60_000
    assert bot._exchange_symbol_unavailable_until_ms == {SYMBOL: refreshed_until}
    assert bot._exchange_config_retry_after_ms == {SYMBOL: refreshed_until}
    assert "action=refresh_entry_block_until_retry" in caplog.text

    modes = {"long": {SYMBOL: None}, "short": {SYMBOL: "panic"}}
    active = bot._apply_exchange_symbol_unavailable_planning_policy(
        [SYMBOL], modes, now_ms=now_ms + 1
    )
    assert active == {SYMBOL}
    assert modes == {
        "long": {SYMBOL: "graceful_stop"},
        "short": {SYMBOL: "panic"},
    }
    assert bot._exchange_symbol_cooldown_blocks_tradability(SYMBOL, active)

    bot.has_position = lambda symbol=None: True
    held_modes = {
        "long": {SYMBOL: "graceful_stop"},
        "short": {SYMBOL: "manual"},
    }
    assert bot._apply_exchange_symbol_unavailable_planning_policy(
        [SYMBOL], held_modes, now_ms=now_ms + 2
    ) == {SYMBOL}
    assert held_modes == {
        "long": {SYMBOL: "tp_only"},
        "short": {SYMBOL: "manual"},
    }
    assert not bot._exchange_symbol_cooldown_blocks_tradability(SYMBOL, active)

    with caplog.at_level("INFO"):
        assert bot._active_exchange_symbol_unavailable_cooldowns(
            [SYMBOL], now_ms=expected_until
        ) == {SYMBOL: refreshed_until}
        assert bot._active_exchange_symbol_unavailable_cooldowns(
            [SYMBOL], now_ms=refreshed_until
        ) == {}
    assert bot._exchange_symbol_unavailable_until_ms == {}
    assert bot._exchange_config_retry_after_ms == {}
    assert bot._exchange_config_retry_attempts == {}
    assert "cooldown expired" in caplog.text

    assert bot._activate_exchange_symbol_unavailable_cooldown(
        SYMBOL, error, now_ms=refreshed_until + 1
    )
    assert bot._exchange_symbol_unavailable_until_ms == {
        SYMBOL: refreshed_until + 1 + 6 * 60 * 60 * 1000
    }


def test_weex_symbol_unavailable_cooldown_can_be_disabled():
    bot = _bot()
    bot.config["live"]["exchange_symbol_unavailable_cooldown_hours"] = 0.0
    error = ccxt.PermissionDenied(
        'weex {"code":-1058,"msg":"The trading pair is not supported via the API"}'
    )

    assert not bot._activate_exchange_symbol_unavailable_cooldown(
        SYMBOL, error, now_ms=1_000_000
    )
    assert getattr(bot, "_exchange_symbol_unavailable_until_ms", {}) == {}


def test_weex_symbol_unavailable_cooldown_rejects_unsafe_runtime_value(caplog):
    bot = _bot()
    bot.config["live"]["exchange_symbol_unavailable_cooldown_hours"] = 1e308

    with caplog.at_level("ERROR"):
        activated = bot._activate_exchange_symbol_unavailable_cooldown(
            SYMBOL,
            ccxt.PermissionDenied(
                'weex {"code":-1058,"msg":"The trading pair is not supported via the API"}'
            ),
            now_ms=1_000,
        )

    assert activated is False
    assert getattr(bot, "_exchange_symbol_unavailable_until_ms", {}) == {}
    assert "action=preserve_original_exchange_failure" in caplog.text


def test_weex_symbol_config_gates_entries_but_not_protective_closes():
    bot = _bot()

    assert bot._order_requires_exchange_config_before_create(
        {"reduce_only": False}
    )
    assert not bot._order_requires_exchange_config_before_create(
        {"reduce_only": True}
    )
    bot._last_exchange_config_failed_attempt_symbols = {SYMBOL}
    assert bot._pending_exchange_config_consumes_error_budget(
        [{"symbol": SYMBOL, "reduce_only": False}]
    )
    bot._last_exchange_config_failed_attempt_symbols = set()
    assert not bot._pending_exchange_config_consumes_error_budget(
        [{"symbol": SYMBOL, "reduce_only": False}]
    )
    assert not bot._pending_exchange_config_consumes_error_budget([])


def test_weex_balance_excludes_unrealized_pnl_from_equity():
    bot = _bot()
    raw = [
        {
            "asset": "USDT",
            "balance": "100.1875",
            "availableBalance": "89.0",
            "frozen": "10.0",
            "unrealizePnl": "-0.25",
        }
    ]
    unified = _ccxt_exchange().parse_balance(raw)

    assert unified["total"]["USDT"] == pytest.approx(100.1875)
    assert bot._get_balance(unified) == pytest.approx(100.4375)


def test_weex_wallet_balance_is_stable_across_mark_to_market_changes():
    bot = _bot()

    def fetched(equity: float, unrealized_pnl: float) -> dict:
        return {
            "info": [
                {
                    "asset": "USDT",
                    "balance": str(equity),
                    "unrealizePnl": str(unrealized_pnl),
                }
            ]
        }

    assert bot._get_balance(fetched(100.2, -0.3)) == pytest.approx(100.5)
    assert bot._get_balance(fetched(100.8, 0.3)) == pytest.approx(100.5)


@pytest.mark.parametrize(
    "info",
    [
        [],
        [{"asset": "BTC", "balance": "1", "unrealizePnl": "0"}],
        [
            {"asset": "USDT", "balance": "1", "unrealizePnl": "0"},
            {"asset": "USDT", "balance": "1", "unrealizePnl": "0"},
        ],
        [{"asset": "USDT", "balance": "1"}],
        [{"asset": "USDT", "balance": "nan", "unrealizePnl": "0"}],
    ],
)
def test_weex_balance_rejects_missing_ambiguous_or_nonfinite_wallet_inputs(info):
    bot = _bot()

    with pytest.raises(ValueError, match="balance response"):
        bot._get_balance({"info": info})


def test_weex_order_websocket_ignores_empty_order_event():
    exchange = ProWeex()
    client = MagicMock()

    assert exchange.handle_orders(client, {"e": "orders", "d": []}) is None
    client.resolve.assert_not_called()


def test_weex_order_websocket_delegates_nonempty_order_event():
    exchange = ProWeex()
    client = MagicMock()
    message = {"e": "orders", "d": [{"symbol": "cmt_btcusdt"}]}

    with patch.object(ccxt.pro.weex, "handle_orders", return_value="handled") as handler:
        assert exchange.handle_orders(client, message) == "handled"
        handler.assert_called_once_with(client, message)


def test_weex_order_websocket_does_not_hide_malformed_order_event():
    exchange = ProWeex()
    client = MagicMock()
    message = {"e": "orders", "d": {"unexpected": "object"}}

    with patch.object(ccxt.pro.weex, "handle_orders", return_value="handled") as handler:
        assert exchange.handle_orders(client, message) == "handled"
        handler.assert_called_once_with(client, message)


def test_weex_order_params_match_v3_hedge_contract():
    bot = _bot(time_in_force="post_only")
    custom_id = "b-WEEX111164-0x0007abc123"
    params = bot._build_order_params(
        {
            "symbol": SYMBOL,
            "type": "limit",
            "position_side": "long",
            "reduce_only": True,
            "custom_id": custom_id,
        }
    )

    assert params == {
        "positionSide": "LONG",
        "clientOrderId": custom_id,
        "timeInForce": "POST_ONLY",
    }
    assert "reduceOnly" not in params
    assert "postOnly" not in params


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_side", ""),
        ("position_side", "both"),
        ("custom_id", ""),
        ("custom_id", "x" * 37),
        ("custom_id", "invalid id"),
        ("custom_id", "0x0004validbutunattributed"),
        ("custom_id", "b-WEEX999999-0x0004wrongbroker"),
        ("custom_id", "b-WEEX111164-without-marker"),
    ],
)
def test_weex_order_params_reject_ambiguous_side_or_invalid_client_id(field, value):
    bot = _bot()
    order = {
        "symbol": SYMBOL,
        "type": "limit",
        "position_side": "long",
        "reduce_only": False,
        "custom_id": "b-WEEX111164-0x0004abc123",
    }
    order[field] = value

    with pytest.raises(ValueError):
        bot._build_order_params(order)


@pytest.mark.parametrize(
    "side,position_side,reduce_only",
    [
        ("buy", "long", False),
        ("sell", "short", False),
        ("sell", "long", True),
        ("buy", "short", True),
    ],
)
def test_weex_order_params_survive_ccxt_request_construction(
    side: str, position_side: str, reduce_only: bool
):
    bot = _bot(time_in_force="post_only")
    custom_id = f"b-WEEX111164-0x0007{position_side}{int(reduce_only)}"
    params = bot._build_order_params(
        {
            "symbol": SYMBOL,
            "type": "limit",
            "position_side": position_side,
            "reduce_only": reduce_only,
            "custom_id": custom_id,
        }
    )
    exchange = _ccxt_exchange()

    request = exchange.create_contract_order_request(
        SYMBOL, "limit", side, 0.0001, 60_000.0, params
    )

    assert request["symbol"] == "BTCUSDT"
    assert request["side"] == side.upper()
    assert request["positionSide"] == position_side.upper()
    assert "positionId" not in request
    assert request["timeInForce"] == "POST_ONLY"
    assert request["newClientOrderId"] == custom_id
    assert "reduceOnly" not in request
    assert "postOnly" not in request


def test_weex_generated_client_id_attributes_broker_and_preserves_diagnostics():
    bot = _bot()

    first = bot.format_custom_id_single(0x0007)
    second = bot.format_custom_id_single(0x0007)

    assert first.startswith("b-WEEX111164-0x0007")
    assert len(first) == 36
    assert first != second
    assert bot.CLIENT_ORDER_ID_PATTERN.fullmatch(first)
    assert custom_id_has_explicit_passivbot_marker(first)
    assert custom_id_to_snake(first) == "close_grid_long"


def test_weex_market_order_keeps_broker_client_id_without_limit_fields():
    bot = _bot()
    custom_id = bot.format_custom_id_single(0x0007)
    params = bot._build_order_params(
        {
            "symbol": SYMBOL,
            "type": "market",
            "position_side": "long",
            "reduce_only": True,
            "custom_id": custom_id,
        }
    )
    request = _ccxt_exchange().create_contract_order_request(
        SYMBOL, "market", "sell", 0.0001, None, params
    )

    assert request["newClientOrderId"] == custom_id
    assert request["positionSide"] == "LONG"
    assert "timeInForce" not in request
    assert "price" not in request
    assert "reduceOnly" not in request


@pytest.mark.parametrize(
    "broker_code", ["", "WEEX11116", "WEEX1111640", "weex111164"]
)
def test_weex_rejects_invalid_broker_code_before_session_creation(broker_code):
    bot = _bot()
    bot.broker_code = broker_code

    with pytest.raises(ValueError, match="WEEX broker code"):
        bot.create_ccxt_sessions()


@pytest.mark.parametrize(
    ("raw_position_side", "expected"),
    [("LONG", "long"), ("SHORT", "short")],
)
def test_weex_open_order_requires_explicit_position_side(
    raw_position_side, expected
):
    bot = _bot()

    assert (
        bot._get_position_side_for_order(
            {"info": {"positionSide": raw_position_side}}
        )
        == expected
    )


@pytest.mark.parametrize("raw_info", [{}, {"positionSide": "BOTH"}, "invalid"])
def test_weex_open_order_rejects_missing_or_ambiguous_position_side(raw_info):
    bot = _bot()

    with pytest.raises(ValueError, match="positionSide|raw info"):
        bot._get_position_side_for_order(
            {
                "clientOrderId": "entry_initial_normal_long",
                "info": raw_info,
            }
        )


def test_weex_signed_order_contains_required_auth_and_no_reduce_only():
    bot = _bot(time_in_force="gtc")
    exchange = _ccxt_exchange()
    custom_id = "b-WEEX111164-0x0007signed01"
    params = bot._build_order_params(
        {
            "symbol": SYMBOL,
            "type": "limit",
            "position_side": "short",
            "reduce_only": True,
            "custom_id": custom_id,
        }
    )
    request = exchange.create_contract_order_request(
        SYMBOL, "limit", "buy", 0.0001, 60_000.0, params
    )

    signed = exchange.sign(
        "capi/v3/order", "contractPrivate", "POST", request
    )
    body = json.loads(signed["body"])

    assert body["positionSide"] == "SHORT"
    assert body["timeInForce"] == "GTC"
    assert body["newClientOrderId"] == custom_id
    assert "reduceOnly" not in body
    assert signed["headers"]["ACCESS-KEY"] == "key"
    assert signed["headers"]["ACCESS-PASSPHRASE"] == "passphrase"
    assert signed["headers"]["ACCESS-SIGN"]


def test_weex_book_ticker_normalization_is_fresh_and_strict():
    bot = _bot()
    bot.markets_dict = {SYMBOL: _market()}
    bot.symbol_ids_inv = {"BTCUSDT": SYMBOL}

    tickers = bot._normalize_tickers(
        [
            {
                "symbol": "BTCUSDT",
                "bidPrice": "60000.1",
                "askPrice": "60000.2",
                "time": 1234,
            },
            {
                "symbol": "BADUSDT",
                "bidPrice": "1",
                "askPrice": "2",
                "time": 1234,
            },
        ]
    )

    assert tickers == {
        SYMBOL: {
            "bid": 60000.1,
            "ask": 60000.2,
            "last": pytest.approx(60000.15),
            "timestamp": 1234,
            "source": "weex_book_ticker_mid",
        }
    }
    assert bot._normalize_tickers(
        [{"symbol": "BTCUSDT", "bidPrice": "2", "askPrice": "1"}]
    ) == {}
    assert bot._normalize_tickers(
        [{"symbol": "BTCUSDT", "bidPrice": "1", "askPrice": "2", "time": 0}]
    ) == {}


@pytest.mark.asyncio
async def test_weex_book_ticker_mid_source_survives_market_snapshot_provider():
    bot = _bot()
    bot.markets_dict = {SYMBOL: _market()}
    bot.symbol_ids_inv = {"BTCUSDT": SYMBOL}

    async def fetch_tickers():
        return bot._normalize_tickers(
            [
                {
                    "symbol": "BTCUSDT",
                    "bidPrice": "99",
                    "askPrice": "101",
                    "time": 1234,
                }
            ]
        )

    provider = MarketSnapshotProvider(
        exchange_name="weex",
        fetch_tickers=fetch_tickers,
        ticker_strategy="bulk",
    )

    snapshot = (await provider.get_snapshots([SYMBOL], max_age_ms=10_000))[SYMBOL]

    assert snapshot.bid == pytest.approx(99.0)
    assert snapshot.ask == pytest.approx(101.0)
    assert snapshot.last == pytest.approx(100.0)
    assert snapshot.source == "weex_book_ticker_mid"


def test_weex_market_sizing_uses_base_quantity_not_contract_val():
    bot = _bot()
    bot.markets_dict = {SYMBOL: _market()}
    bot.eligible_symbols = {SYMBOL}

    bot.set_market_specific_settings()

    assert bot.min_qtys[SYMBOL] == pytest.approx(0.0001)
    assert bot.qty_steps[SYMBOL] == pytest.approx(0.0001)
    assert bot.c_mults[SYMBOL] == 1.0
    assert bot.max_leverage[SYMBOL] == 400


@pytest.mark.asyncio
async def test_weex_symbol_config_sets_combined_margin_and_leverage():
    class _Api:
        def __init__(self):
            self.calls = []

        async def fetch_position_mode(self, symbol):
            self.calls.append(("fetch_position_mode", symbol))
            return {
                "hedged": True,
                "info": [
                    {"symbol": "BTCUSDT", "separatedType": "SEPARATED"}
                ],
            }

        async def fetch_margin_mode(self, symbol):
            self.calls.append(("fetch_margin_mode", symbol))
            return {"marginMode": "isolated"}

        async def set_position_mode(self, hedged, symbol=None, params=None):
            self.calls.append(("set_position_mode", hedged, symbol, params))
            return {"code": "200", "msg": "success"}

        async def set_leverage(self, leverage, symbol=None, params=None):
            self.calls.append(("set_leverage", leverage, symbol, params))
            return {"code": "200", "msg": "success"}

    bot = _bot()
    bot.cca = _Api()
    bot.markets_dict = {SYMBOL: _market()}
    bot.max_leverage = {SYMBOL: 400}
    bot._get_margin_mode_for_symbol = lambda symbol: "cross"
    bot._calc_leverage_for_symbol = lambda symbol: 10

    await bot.update_exchange_config_by_symbols([SYMBOL])

    assert ("set_position_mode", False, SYMBOL, {"marginMode": "cross"}) in bot.cca.calls
    assert ("set_leverage", 10, SYMBOL, {"marginMode": "cross"}) in bot.cca.calls


@pytest.mark.asyncio
async def test_weex_symbol_config_skips_unchanged_mode_but_sets_leverage():
    class _Api:
        def __init__(self):
            self.calls = []

        async def fetch_position_mode(self, symbol):
            return {
                "hedged": False,
                "info": [
                    {"symbol": "BTCUSDT", "separatedType": "COMBINED"}
                ],
            }

        async def fetch_margin_mode(self, symbol):
            return {"marginMode": "cross"}

        async def set_position_mode(self, *args, **kwargs):
            raise AssertionError("unchanged WEEX mode must not be rewritten")

        async def set_leverage(self, leverage, symbol=None, params=None):
            self.calls.append((leverage, symbol, params))
            return {"code": "200", "msg": "success"}

    bot = _bot()
    bot.cca = _Api()
    bot.markets_dict = {SYMBOL: _market()}
    bot._get_margin_mode_for_symbol = lambda symbol: "cross"
    bot._calc_leverage_for_symbol = lambda symbol: 7

    await bot.update_exchange_config_by_symbols([SYMBOL])

    assert bot.cca.calls == [(7, SYMBOL, {"marginMode": "cross"})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "position_mode",
    [
        {"hedged": False, "info": []},
        {
            "hedged": False,
            "info": [{"symbol": "BTCUSDT", "separatedType": "UNKNOWN"}],
        },
        {
            "hedged": False,
            "info": [{"symbol": "ETHUSDT", "separatedType": "COMBINED"}],
        },
        {
            "hedged": False,
            "info": [{"symbol": "BTCUSDT", "separatedType": "SEPARATED"}],
        },
    ],
)
async def test_weex_symbol_config_rejects_unproven_or_inconsistent_position_mode(
    position_mode,
):
    class _Api:
        def __init__(self):
            self.leverage_calls = []

        async def fetch_position_mode(self, symbol):
            return position_mode

        async def fetch_margin_mode(self, symbol):
            return {"marginMode": "cross"}

        async def set_position_mode(self, *args, **kwargs):
            raise AssertionError("invalid mode state must fail before mutation")

        async def set_leverage(self, *args, **kwargs):
            self.leverage_calls.append((args, kwargs))

    bot = _bot()
    bot.cca = _Api()
    bot.markets_dict = {SYMBOL: _market()}
    bot._get_margin_mode_for_symbol = lambda symbol: "cross"
    bot._calc_leverage_for_symbol = lambda symbol: 7

    with pytest.raises(ValueError, match="position mode"):
        await bot.update_exchange_config_by_symbols([SYMBOL])

    assert bot.cca.leverage_calls == []


class _FakeWeexApi:
    def __init__(self, trades, *, newest_first=False):
        self.trades = list(trades)
        self.newest_first = bool(newest_first)
        self.trade_calls = []
        self.order_calls = []

    async def fetch_my_trades(self, symbol=None, since=None, limit=None, params=None):
        self.trade_calls.append(
            {"symbol": symbol, "since": since, "limit": limit, "params": dict(params or {})}
        )
        until = int((params or {})["until"])
        eligible = [
            trade
            for trade in self.trades
            if int(since) <= int(trade["timestamp"]) <= until
        ]
        eligible.sort(
            key=lambda trade: int(trade["timestamp"]),
            reverse=self.newest_first,
        )
        return eligible[: int(limit)]

    async def fetch_order(self, order_id, symbol):
        self.order_calls.append((order_id, symbol))
        return {
            "clientOrderId": f"b-WEEX111164-0x0004{order_id}",
            "info": {},
        }


def _trade(trade_id: str, order_id: str, timestamp: int, *, position_side="LONG"):
    return {
        "id": trade_id,
        "order": order_id,
        "timestamp": timestamp,
        "symbol": SYMBOL,
        "side": "buy",
        "amount": 0.0001,
        "price": 60_000.0,
        "fee": {"currency": "USDT", "cost": 0.0012},
        "info": {
            "id": trade_id,
            "orderId": order_id,
            "time": timestamp,
            "side": "BUY",
            "positionSide": position_side,
            "realizedPnl": "1.25",
        },
    }


@pytest.mark.asyncio
async def test_weex_fetcher_windows_paginates_normalizes_and_enriches():
    day = 24 * 60 * 60 * 1000
    api = _FakeWeexApi(
        [
            _trade("t1", "o1", day),
            _trade("t2", "o2", day + 1),
            _trade("t3", "o3", day + 2),
            _trade("t4", "o4", 8 * day),
        ],
        newest_first=True,
    )
    fetcher = WeexFetcher(api, trade_limit=2, now_func=lambda: 9 * day)
    detail_cache = {}

    events = await fetcher.fetch(day, 9 * day, detail_cache)

    assert [event["id"] for event in events] == ["t1", "t2", "t3", "t4"]
    assert all(event["position_side"] == "long" for event in events)
    assert all(event["pnl"] == pytest.approx(1.25) for event in events)
    assert all(event["c_mult"] == 1.0 for event in events)
    assert all(
        event["client_order_id"].startswith("b-WEEX111164-0x0004")
        for event in events
    )
    assert all(event["pb_order_type"] == "entry_grid_normal_long" for event in events)
    assert len(api.order_calls) == 4
    assert all(call["params"]["type"] == "swap" for call in api.trade_calls)
    assert all(
        call["params"]["until"] - call["since"] < WeexFetcher.WINDOW_MS
        for call in api.trade_calls
    )
    assert set(detail_cache) == {"t1", "t2", "t3", "t4"}


@pytest.mark.asyncio
async def test_weex_fetcher_fails_closed_when_one_millisecond_is_saturated():
    api = _FakeWeexApi(
        [_trade("t1", "o1", 1_000), _trade("t2", "o2", 1_000)]
    )
    fetcher = WeexFetcher(api, trade_limit=2)

    with pytest.raises(RuntimeError, match="saturated within one millisecond"):
        await fetcher.fetch(1_000, 1_000, {})


def test_weex_fetcher_rejects_ambiguous_position_side():
    with pytest.raises(ValueError, match="positionSide"):
        WeexFetcher._normalize_trade(
            _trade("t1", "o1", 1, position_side="")
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("amount", 0.0, "non-positive"),
        ("price", 0.0, "non-positive"),
        ("realizedPnl", None, "realizedPnl"),
    ],
)
def test_weex_fetcher_rejects_invalid_accounting_values(field, value, match):
    trade = _trade("t1", "o1", 1)
    if field == "realizedPnl":
        trade["info"].pop(field)
    else:
        trade[field] = value
        trade["info"]["qty" if field == "amount" else field] = value

    with pytest.raises(ValueError, match=match):
        WeexFetcher._normalize_trade(trade)


def test_build_fetcher_for_weex():
    bot = SimpleNamespace(
        exchange="weex",
        cca="api",
        user="weex_user",
        markets_dict={},
        coin_to_symbol=lambda value, verbose=False: value,
    )

    assert isinstance(_build_fetcher_for_bot(bot, symbols=["BTC"]), WeexFetcher)


def test_setup_bot_weex_uses_weex_adapter():
    from passivbot import setup_bot

    config = {"live": {"user": "weex_01"}}
    with patch("passivbot.load_user_info", return_value={"exchange": "weex"}):
        with patch("exchanges.weex.WeexBot") as mock_cls:
            mock_bot = MagicMock()
            mock_cls.return_value = mock_bot
            assert setup_bot(config) is mock_bot
            mock_cls.assert_called_once_with(config)
            assert mock_bot._order_churn_gate_enabled_for_connector is True
