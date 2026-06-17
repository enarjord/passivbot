from types import SimpleNamespace

import ccxt
import pytest

from ccxt_contracts import build_contract_bot
from exchanges.bitget import BitgetBot, deduce_uta_side_pside
from exchanges.binance import BinanceBot
from exchanges.bybit import BybitBot
from exchanges.defx import DefxBot
from exchanges.gateio import GateIOBot
from exchanges.kucoin import KucoinBot
from exchanges.okx import OKXBot
from passivbot import custom_id_has_explicit_passivbot_marker, custom_id_to_snake
from passivbot_exceptions import FatalBotException
from pure_funcs import determine_pos_side_ccxt


def test_determine_pos_side_ccxt_handles_common_ccxt_order_variants():
    assert determine_pos_side_ccxt({"info": {"positionIdx": 1}}) == "long"
    assert determine_pos_side_ccxt({"info": {"positionIdx": 2}}) == "short"
    assert determine_pos_side_ccxt({"info": {"positionSide": "SHORT"}}) == "short"
    assert determine_pos_side_ccxt({"info": {"side": "buy", "reduceOnly": False}}) == "long"
    assert determine_pos_side_ccxt({"info": {"side": "buy", "reduceOnly": True}}) == "short"
    assert determine_pos_side_ccxt({"info": {"side": "buy", "closedSize": "0"}}) == "short"
    assert determine_pos_side_ccxt({"info": {"orderLinkId": "entry_long_01"}}) == "long"
    assert determine_pos_side_ccxt({"info": {"clientOid": "close_shrt_02"}}) == "short"


def test_ccxt_bot_build_order_params_tracks_ccxt_parameter_names():
    bot = build_contract_bot("testexchange")
    params = bot._build_order_params(
        {
            "type": "limit",
            "position_side": "long",
            "custom_id": "entry_long_01",
        }
    )
    assert params == {
        "positionSide": "LONG",
        "clientOrderId": "entry_long_01",
        "timeInForce": "GTC",
    }


def test_binance_position_side_uses_raw_info_ps_field():
    bot = BinanceBot.__new__(BinanceBot)
    assert bot._get_position_side_for_order({"info": {"ps": "SHORT"}}) == "short"


def test_binance_open_order_symbol_selection_can_seed_from_approved_when_empty():
    bot = BinanceBot.__new__(BinanceBot)
    bot.open_orders = {}
    bot.positions = {}
    bot.active_symbols = []
    bot.approved_coins_minus_ignored_coins = {
        "long": {"ADA/USDT:USDT"},
        "short": {"SOL/USDT:USDT"},
    }
    bot.markets_dict = {"ADA/USDT:USDT": {}, "SOL/USDT:USDT": {}}

    assert bot._select_open_order_symbols(include_approved_if_unseeded=True) == {
        "ADA/USDT:USDT",
        "SOL/USDT:USDT",
    }


def test_binance_normalize_positions_blocks_dated_futures_position():
    bot = BinanceBot.__new__(BinanceBot)
    bot.markets_dict = {
        "BTC_260327/USDT:USDT": {
            "id": "BTCUSDT_260327",
            "future": True,
            "swap": False,
            "expiry": 1774579200000,
        }
    }

    with pytest.raises(FatalBotException, match="Unsupported dated futures position"):
        bot._normalize_positions(
            [
                {
                    "symbol": "BTCUSDT_260327",
                    "positionAmt": "0.01",
                    "positionSide": "LONG",
                    "entryPrice": "50000",
                }
            ]
        )


@pytest.mark.asyncio
async def test_binance_fetch_tickers_skips_dated_futures_before_symbol_conversion():
    bot = BinanceBot.__new__(BinanceBot)
    bot.markets_dict = {
        "BTC/USDT:USDT": {"id": "BTCUSDT", "swap": True},
        "BTC_260327/USDT:USDT": {
            "id": "BTCUSDT_260327",
            "future": True,
            "swap": False,
            "expiry": 1774579200000,
        },
    }
    bot.symbol_ids_inv = {"BTCUSDT": "BTC/USDT:USDT"}

    class _Api:
        async def fapipublic_get_ticker_bookticker(self):
            return [
                {"symbol": "BTCUSDT_260327", "bidPrice": "1", "askPrice": "2"},
                {"symbol": "BTCUSDT", "bidPrice": "100", "askPrice": "101"},
            ]

    bot.cca = _Api()

    tickers = await bot.fetch_tickers()

    assert sorted(tickers) == ["BTC/USDT:USDT"]


@pytest.mark.asyncio
async def test_binance_staged_snapshot_uses_fresh_positions_for_open_order_symbols():
    bot = BinanceBot.__new__(BinanceBot)
    bot.exchange = "binance"
    bot.open_orders = {}
    bot.positions = {}
    bot.active_symbols = []
    bot.approved_coins_minus_ignored_coins = {"long": set(), "short": set()}
    bot.markets_dict = {"SOL/USDT:USDT": {}}
    bot._record_live_margin_mode_from_payload = lambda payload: None

    async def capture_balance_snapshot():
        return {"raw": "balance"}, 100.0

    async def capture_positions_snapshot():
        return [{"raw": "position"}], [
            {
                "symbol": "SOL/USDT:USDT",
                "position_side": "long",
                "size": 1.0,
                "price": 100.0,
            }
        ]

    seen_fill_scope = []

    async def update_pnls(**_kwargs):
        seen_fill_scope.append(list(bot._fill_symbol_scope))
        return True

    fetched_symbols = []

    async def fetch_open_orders(symbol=None):
        fetched_symbols.append(symbol)
        return [
            {
                "id": "order-1",
                "symbol": symbol,
                "amount": 0.1,
                "timestamp": 1,
                "info": {"positionSide": "LONG"},
            }
        ]

    bot.capture_balance_snapshot = capture_balance_snapshot
    bot.capture_positions_snapshot = capture_positions_snapshot
    bot.update_pnls = update_pnls
    bot.cca = SimpleNamespace(fetch_open_orders=fetch_open_orders)

    snapshot = await bot.capture_authoritative_state_staged_snapshot(
        {"balance", "positions", "open_orders", "fills"}, {}
    )

    assert fetched_symbols == ["SOL/USDT:USDT"]
    assert snapshot["balance"] == 100.0
    assert snapshot["positions"][0]["symbol"] == "SOL/USDT:USDT"
    assert snapshot["open_orders"][0]["symbol"] == "SOL/USDT:USDT"
    assert snapshot["open_orders"][0]["position_side"] == "long"
    assert seen_fill_scope == [["SOL/USDT:USDT"]]
    assert not hasattr(bot, "_fill_symbol_scope")


def test_bybit_position_side_uses_determine_pos_side_ccxt_contract():
    bot = BybitBot.__new__(BybitBot)
    assert bot._get_position_side_for_order({"info": {"positionIdx": 2}}) == "short"
    assert bot._get_position_side_for_order({"info": {"positionSide": "LONG"}}) == "long"


@pytest.mark.asyncio
async def test_bybit_unified_balance_uses_equity_minus_perp_upl():
    bot = BybitBot.__new__(BybitBot)
    bot.exchange = "bybit"
    bot.quote = "USDT"

    fetched = {
        "total": {"BTC": 0.01654816, "USDT": -130.46645492},
        "info": {
            "result": {
                "list": [
                    {
                        "accountType": "UNIFIED",
                        "totalEquity": "1019.90778334",
                        "totalPerpUPL": "-51.42503948",
                        "coin": [
                            {
                                "coin": "BTC",
                                "marginCollateral": True,
                                "collateralSwitch": True,
                                "walletBalance": "0.01654816",
                                "usdValue": "1310.47850227",
                                "unrealisedPnl": "0",
                            },
                            {
                                "coin": "USDT",
                                "marginCollateral": True,
                                "collateralSwitch": True,
                                "walletBalance": "-130.46645492",
                                "usdValue": "-290.57071892",
                                "unrealisedPnl": "-51.4485",
                            },
                        ],
                    }
                ]
            }
        },
    }

    async def _fetch_balance():
        return fetched

    bot.cca = SimpleNamespace(fetch_balance=_fetch_balance)

    legacy_balance = await bot.fetch_balance()
    _raw_snapshot, staged_balance = await bot.capture_balance_snapshot()

    assert legacy_balance == pytest.approx(1071.33282282)
    assert staged_balance == pytest.approx(legacy_balance)


def test_bybit_unified_balance_falls_back_to_collateral_equity_without_upnl():
    bot = BybitBot.__new__(BybitBot)
    bot.exchange = "bybit"
    bot.quote = "USDT"

    fetched = {
        "total": {"USDT": 10.0},
        "info": {
            "result": {
                "list": [
                    {
                        "accountType": "UNIFIED",
                        "coin": [
                            {
                                "coin": "BTC",
                                "marginCollateral": "true",
                                "collateralSwitch": "true",
                                "usdValue": "1310.0",
                                "unrealisedPnl": "0",
                            },
                            {
                                "coin": "USDT",
                                "marginCollateral": True,
                                "collateralSwitch": True,
                                "usdValue": "-290.0",
                                "unrealisedPnl": "-50.0",
                            },
                        ],
                    }
                ]
            }
        },
    }

    assert bot._get_balance(fetched) == pytest.approx(1070.0)


def test_bitget_determine_side_uses_trade_side_reduce_only_and_pos_side():
    bot = BitgetBot.__new__(BitgetBot)
    close_short = {
        "info": {"tradeSide": "close", "reduceOnly": True, "posSide": "short"},
    }
    open_long = {
        "info": {"tradeSide": "open", "reduceOnly": False, "posSide": "long"},
    }
    assert bot._determine_side(close_short) == "buy"
    assert bot._determine_side(open_long) == "buy"


def test_bitget_uta_custom_id_respects_v3_client_oid_contract():
    bot = BitgetBot.__new__(BitgetBot)
    bot.broker_code = "p4sve"
    bot.custom_id_max_length = 64
    bot.is_uta = True

    custom_id = bot.format_custom_id_single(0x0007)

    allowed = set(".ABCDEFGHIJKLMNOPQRSTUVWXYZ:/abcdefghijklmnopqrstuvwxyz0123456789_-")
    assert len(custom_id) <= 32
    assert "#" not in custom_id
    assert set(custom_id) <= allowed
    assert custom_id_has_explicit_passivbot_marker(custom_id)
    assert custom_id_to_snake(custom_id) != "unknown"


def test_bitget_uta_order_params_use_ccxt_v3_safe_time_in_force():
    bot = BitgetBot.__new__(BitgetBot)
    bot.config = {"live": {"time_in_force": "post_only"}}
    bot.is_uta = True
    order = {
        "position_side": "short",
        "reduce_only": True,
        "custom_id": "0x0007abcdef",
    }

    post_only_params = bot._build_order_params(order)

    assert post_only_params["postOnly"] is True
    assert "timeInForce" not in post_only_params
    assert post_only_params["posSide"] == "short"
    assert post_only_params["reduceOnly"] is True
    assert post_only_params["clientOid"] == "0x0007abcdef"

    bot.config = {"live": {"time_in_force": "gtc"}}
    gtc_params = bot._build_order_params(order)

    assert gtc_params["timeInForce"] == "gtc"
    assert "postOnly" not in gtc_params


def test_bitget_uta_order_params_survive_ccxt_v3_request_construction():
    bot = BitgetBot.__new__(BitgetBot)
    bot.config = {"live": {"time_in_force": "gtc"}}
    bot.is_uta = True
    params = bot._build_order_params(
        {
            "position_side": "short",
            "reduce_only": True,
            "custom_id": "0x0007abcdef",
        }
    )

    exchange = ccxt.bitget()
    exchange.markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "symbol": "BTC/USDT:USDT",
            "spot": False,
            "swap": True,
            "linear": True,
            "type": "swap",
            "contract": True,
            "settle": "USDT",
        }
    }
    exchange.market = lambda symbol: exchange.markets[symbol]
    exchange.handle_product_type_and_params = lambda market, params: ("USDT-FUTURES", params)
    exchange.amount_to_precision = lambda symbol, amount: str(amount)
    exchange.price_to_precision = lambda symbol, price: str(price)

    request = exchange.create_uta_order_request(
        "BTC/USDT:USDT", "limit", "buy", 0.1, 9.0, params
    )

    assert request["timeInForce"] == "gtc"
    assert request["reduceOnly"] == "yes"
    assert request["posSide"] == "short"
    assert request["clientOid"] == "0x0007abcdef"


def test_bitget_uta_broker_code_is_signed_as_v3_channel_header():
    exchange = ccxt.bitget({"apiKey": "key", "secret": "secret", "password": "passphrase"})
    exchange.options["broker"] = "p4sve"

    signed = exchange.sign(
        "v3/trade/place-order",
        ["private", "uta"],
        "POST",
        {
            "category": "USDT-FUTURES",
            "symbol": "BTCUSDT",
            "qty": "0.1",
            "side": "buy",
            "orderType": "limit",
            "price": "9",
            "timeInForce": "gtc",
        },
    )

    assert signed["headers"]["X-CHANNEL-API-CODE"] == "p4sve"


def test_bitget_uta_balance_uses_equity_minus_upnl_not_effective_margin():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = True
    bot.quote = "USDT"

    balance = bot._get_balance(
        {
            "data": {
                "effEquity": "941.70",
                "usdtEquity": "1945.22",
                "accountEquity": "1943.49",
                "usdtUnrealisedPnl": "-12.30",
                "assets": [
                    {"coin": "BTC", "balance": "0.015"},
                    {"coin": "USDT", "balance": "311.51", "available": "311.51"},
                ],
            }
        }
    )

    assert balance == pytest.approx(1957.52)


def test_bitget_uta_balance_rejects_missing_account_equity_fields():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = True
    bot.quote = "USDT"

    with pytest.raises(ValueError, match="account equity/upnl"):
        bot._get_balance(
            {
                "data": {
                    "effEquity": "941.70",
                    "assets": [{"coin": "USDT", "balance": "311.51", "available": "311.51"}],
                }
            }
        )


@pytest.mark.asyncio
async def test_bitget_uta_fetch_balance_uses_raw_account_assets_endpoint():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = True

    class _Api:
        def __init__(self):
            self.calls = []

        async def private_uta_get_v3_account_assets(self):
            self.calls.append("private_uta_get_v3_account_assets")
            return {"data": {"effEquity": "123.45"}}

        async def fetch_balance(self):
            raise AssertionError("CCXT parsed fetch_balance should not be used for UTA")

    api = _Api()
    bot.cca = api

    fetched = await bot._do_fetch_balance()

    assert fetched == {"data": {"effEquity": "123.45"}}
    assert api.calls == ["private_uta_get_v3_account_assets"]


@pytest.mark.asyncio
async def test_bitget_account_mode_detection_sets_uta_on_v3_success():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = False
    bot._account_mode_detected = False

    async def private_uta_get_v3_account_assets():
        return {"data": {"usdtEquity": "10", "usdtUnrealisedPnl": "0"}}

    bot.cca = SimpleNamespace(options={}, private_uta_get_v3_account_assets=private_uta_get_v3_account_assets)
    bot.ccp = SimpleNamespace(options={})

    await bot._detect_account_mode()

    assert bot.is_uta is True
    assert bot._account_mode_detected is True
    assert bot.cca.options["uta"] is True
    assert bot.ccp.options["uta"] is True


@pytest.mark.asyncio
async def test_bitget_account_mode_detection_uses_classic_only_for_explicit_code():
    for code in ("40084", "25245"):
        bot = BitgetBot.__new__(BitgetBot)
        bot.is_uta = True
        bot._account_mode_detected = False

        async def private_uta_get_v3_account_assets(error_code=code):
            raise Exception(f'bitget {{"code":"{error_code}","msg":"Classic Account mode"}}')

        bot.cca = SimpleNamespace(
            options={},
            private_uta_get_v3_account_assets=private_uta_get_v3_account_assets,
        )
        bot.ccp = SimpleNamespace(options={})

        await bot._detect_account_mode()

        assert bot.is_uta is False
        assert bot._account_mode_detected is True
        assert bot.cca.options["uta"] is False
        assert bot.ccp.options["uta"] is False


@pytest.mark.asyncio
async def test_bitget_account_mode_detection_rejects_inconclusive_errors():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = False
    bot._account_mode_detected = False

    async def private_uta_get_v3_account_assets():
        raise TimeoutError("temporary network failure")

    bot.cca = SimpleNamespace(options={}, private_uta_get_v3_account_assets=private_uta_get_v3_account_assets)
    bot.ccp = SimpleNamespace(options={})

    with pytest.raises(RuntimeError, match="inconclusively"):
        await bot._detect_account_mode()

    assert bot._account_mode_detected is False
    assert "uta" not in bot.cca.options


def test_bitget_uta_fill_side_pside_uses_trade_side_when_pos_side_absent():
    assert deduce_uta_side_pside({"side": "sell", "tradeSide": "open"}) == ("sell", "short")
    assert deduce_uta_side_pside({"side": "buy", "tradeSide": "close"}) == ("buy", "short")
    assert deduce_uta_side_pside({"side": "sell", "tradeSide": "close"}) == ("sell", "long")
    assert deduce_uta_side_pside({"side": "buy", "posSide": "long"}) == ("buy", "long")
    with pytest.raises(ValueError, match="order side"):
        deduce_uta_side_pside({"posSide": "long"})


@pytest.mark.asyncio
async def test_bitget_fetch_fill_events_uta_uses_exec_id_and_trade_side():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = True
    bot.get_symbol_id_inv = lambda symbol: "BTC/USDT:USDT" if symbol == "BTCUSDT" else symbol
    calls = []

    async def private_uta_get_v3_trade_fills(params):
        calls.append(dict(params))
        if len(calls) > 1:
            return {"data": {"list": []}}
        return {
            "data": {
                "list": [
                    {
                        "execId": "exec-close-short",
                        "orderId": "order-close-short",
                        "clientOid": "0x0007abcdef",
                        "createdTime": "1000",
                        "symbol": "BTCUSDT",
                        "side": "buy",
                        "tradeSide": "close",
                        "execQty": "0.1",
                        "execPrice": "9",
                        "execPnl": "1",
                        "feeDetail": [{"feeCoin": "USDT", "fee": "0.01"}],
                    }
                ]
            }
        }

    bot.cca = SimpleNamespace(private_uta_get_v3_trade_fills=private_uta_get_v3_trade_fills)

    events = await bot._fetch_fill_events_uta(start_time=900, end_time=2000, limit=100)

    assert len(events) == 1
    assert events[0]["id"] == "exec-close-short"
    assert events[0]["order_id"] == "order-close-short"
    assert events[0]["position_side"] == "short"
    assert events[0]["side"] == "buy"
    assert events[0]["symbol"] == "BTC/USDT:USDT"
    assert events[0]["fees"][0]["fee_paid"] == pytest.approx(-0.01)


@pytest.mark.asyncio
async def test_bitget_fetch_fill_events_uta_rejects_malformed_rows():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = True
    bot.get_symbol_id_inv = lambda symbol: symbol

    async def private_uta_get_v3_trade_fills(params):
        return {
            "data": {
                "list": [
                    {
                        "execId": "exec-bad",
                        "orderId": "order-bad",
                        "symbol": "BTCUSDT",
                        "side": "buy",
                        "tradeSide": "open",
                        "execQty": "0.1",
                        "execPrice": "9",
                        "execPnl": "0",
                    }
                ]
            }
        }

    bot.cca = SimpleNamespace(private_uta_get_v3_trade_fills=private_uta_get_v3_trade_fills)

    with pytest.raises(ValueError, match="createdTime"):
        await bot._fetch_fill_events_uta(start_time=900, end_time=2000, limit=100)


@pytest.mark.asyncio
async def test_bitget_fetch_closed_orders_routes_uta_to_v3_fills():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = True
    bot.get_symbol_id_inv = lambda symbol: "BTC/USDT:USDT" if symbol == "BTCUSDT" else symbol
    calls = []

    async def private_uta_get_v3_trade_fills(params):
        calls.append(dict(params))
        if len(calls) > 1:
            return {"data": {"list": []}}
        return {
            "data": {
                "list": [
                    {
                        "execId": "exec-close-long",
                        "orderId": "order-close-long",
                        "createdTime": "1000",
                        "symbol": "BTCUSDT",
                        "side": "sell",
                        "tradeSide": "close",
                        "execQty": "0.1",
                        "execPrice": "11",
                        "execPnl": "1",
                    }
                ]
            }
        }

    async def fetch_closed_orders(*args, **kwargs):
        raise AssertionError("UTA closed-order history must use v3 trade fills")

    bot.cca = SimpleNamespace(
        private_uta_get_v3_trade_fills=private_uta_get_v3_trade_fills,
        fetch_closed_orders=fetch_closed_orders,
    )

    events = await bot.fetch_closed_orders(start_time=900, end_time=2000, limit=100)

    assert len(events) == 1
    assert events[0]["id"] == "exec-close-long"
    assert events[0]["pnl"] == pytest.approx(1.0)
    assert events[0]["position_side"] == "long"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("info", "match"),
    [
        ({"posSide": "long"}, "totalProfits"),
        ({"totalProfits": "1.0"}, "posSide"),
    ],
)
async def test_bitget_fetch_closed_orders_requires_pnl_and_position_side(info, match):
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = False

    async def fetch_closed_orders(limit=None, params=None):
        return [
            {
                "id": "closed-order",
                "lastUpdateTimestamp": "1000",
                "timestamp": 1000,
                "symbol": "BTC/USDT:USDT",
                "side": "sell",
                "price": "11",
                "filled": "0.1",
                "clientOrderId": "0x0007abcdef",
                "info": info,
            }
        ]

    bot.cca = SimpleNamespace(fetch_closed_orders=fetch_closed_orders)

    with pytest.raises(ValueError, match=match):
        await bot.fetch_closed_orders(start_time=900, end_time=2000, limit=100)


def test_gateio_order_side_contract_depends_on_reduce_only_flag():
    bot = GateIOBot.__new__(GateIOBot)
    assert bot.determine_pos_side({"side": "buy", "reduceOnly": False}) == "long"
    assert bot.determine_pos_side({"side": "buy", "reduceOnly": True}) == "short"
    assert bot.determine_pos_side({"side": "sell", "reduceOnly": False}) == "short"
    assert bot.determine_pos_side({"side": "sell", "reduceOnly": True}) == "long"


def test_gateio_order_params_use_client_order_id_for_ccxt_text_prefix():
    bot = GateIOBot.__new__(GateIOBot)
    bot.config = {"live": {"time_in_force": "post_only"}}
    params = bot._build_order_params(
        {
            "type": "limit",
            "reduce_only": False,
            "custom_id": "0x0004abcdef",
        }
    )

    assert params["clientOrderId"] == "0x0004abcdef"
    assert "text" not in params
    assert params["reduce_only"] is False
    assert params["timeInForce"] == "poc"

    gateio_text = "t-" + params["clientOrderId"]
    assert custom_id_has_explicit_passivbot_marker(gateio_text)
    assert custom_id_to_snake(gateio_text) == "entry_grid_normal_long"


def test_gateio_get_balance_uses_cross_available_for_multi_currency_margin():
    bot = GateIOBot.__new__(GateIOBot)
    bot.exchange = "gateio"
    bot.quote = "USDT"
    bot.uid = None
    bot.cca = SimpleNamespace()
    bot.ccp = SimpleNamespace()
    bot.log_once = lambda msg: None

    balance = bot._get_balance(
        {
            "USDT": {"total": 0.000000000035},
            "info": [
                {
                    "user": "16770081",
                    "margin_mode_name": "multi_currency",
                    "cross_available": "724.95615",
                }
            ],
        }
    )

    assert balance == 724.95615
    assert bot.uid == "16770081"
    assert bot.cca.uid == "16770081"
    assert bot.ccp.uid == "16770081"


def test_gateio_get_balance_uses_total_for_classic_margin():
    bot = GateIOBot.__new__(GateIOBot)
    bot.exchange = "gateio"
    bot.quote = "USDT"
    bot.uid = "existing"
    bot.cca = SimpleNamespace(uid="existing")
    bot.ccp = None
    bot.log_once = lambda msg: None

    balance = bot._get_balance(
        {
            "USDT": {"total": 543.21},
            "info": [
                {
                    "user": "16770081",
                    "margin_mode_name": "classic",
                    "cross_available": "724.95615",
                }
            ],
        }
    )

    assert balance == 543.21


def test_okx_order_side_uses_info_pos_side():
    bot = OKXBot.__new__(OKXBot)
    assert bot._get_position_side_for_order({"info": {"posSide": "SHORT"}}) == "short"


def test_okx_get_balance_includes_multi_asset_collateral_excluding_upl():
    bot = OKXBot.__new__(OKXBot)
    bot.exchange = "okx"
    bot.quote = "USDT"

    balance = bot._get_balance(
        {
            "total": {"BTC": 0.0028398211190601, "USDT": 68.18073094671803},
            "info": {
                "data": [
                    {
                        "totalEq": "285.6112921470932",
                        "details": [
                            {
                                "ccy": "BTC",
                                "cashBal": "0.0028398211190601",
                                "eqUsd": "217.4360156588509",
                                "upl": "0",
                                "collateralEnabled": True,
                            },
                            {
                                "ccy": "USDT",
                                "cashBal": "76.92702623479187",
                                "eqUsd": "68.17527648824229",
                                "upl": "-8.746295288073837",
                                "collateralEnabled": True,
                            },
                        ],
                    }
                ]
            },
        }
    )

    assert balance == pytest.approx(294.35758743516704)


@pytest.mark.asyncio
async def test_okx_legacy_and_staged_balance_snapshot_use_same_collateral_value():
    bot = OKXBot.__new__(OKXBot)
    bot.exchange = "okx"
    bot.quote = "USDT"

    fetched = {
        "total": {"BTC": 0.0028398211190601, "USDT": 68.18073094671803},
        "info": {
            "data": [
                {
                    "totalEq": "285.6112921470932",
                    "details": [
                        {
                            "ccy": "BTC",
                            "cashBal": "0.0028398211190601",
                            "eqUsd": "217.4360156588509",
                            "upl": "0",
                            "collateralEnabled": True,
                        },
                        {
                            "ccy": "USDT",
                            "cashBal": "76.92702623479187",
                            "eqUsd": "68.17527648824229",
                            "upl": "-8.746295288073837",
                            "collateralEnabled": True,
                        },
                    ],
                }
            ]
        },
    }

    async def _fetch_balance():
        return fetched

    bot.cca = SimpleNamespace(fetch_balance=_fetch_balance)

    legacy_balance = await bot.fetch_balance()
    _raw_snapshot, staged_balance = await bot.capture_balance_snapshot()

    assert legacy_balance == pytest.approx(294.35758743516704)
    assert staged_balance == pytest.approx(legacy_balance)


def test_one_way_exchanges_prefer_existing_position_state_over_side():
    for bot_cls in (DefxBot, KucoinBot):
        bot = bot_cls.__new__(bot_cls)
        bot.positions = {
            "BTC/USDT:USDT": {
                "long": {"size": 1.0},
                "short": {"size": 0.0},
            }
        }
        bot.has_position = lambda pside, symbol: abs(
            float(bot.positions.get(symbol, {}).get(pside, {}).get("size", 0.0) or 0.0)
        ) > 0.0
        assert bot.determine_pos_side({"symbol": "BTC/USDT:USDT", "side": "sell"}) == "long"


@pytest.mark.asyncio
async def test_bitget_fetch_open_orders_preserves_raw_pos_side_and_client_order_id():
    bot = BitgetBot.__new__(BitgetBot)
    bot.quote = "USDT"
    bot._live_margin_modes = {}
    bot._record_live_margin_mode_from_payload = lambda payload: None
    bot.cca = SimpleNamespace()

    async def _fetch_open_orders(symbol=None):
        return [
            {
                "id": "1",
                "symbol": "BTC/USDT:USDT",
                "amount": 0.5,
                "timestamp": 10,
                "clientOrderId": "entry_long_01",
                "info": {"tradeSide": "open", "reduceOnly": False, "posSide": "long"},
            }
        ]

    bot.cca.fetch_open_orders = _fetch_open_orders
    orders = await bot.fetch_open_orders(symbol="BTC/USDT:USDT")

    assert orders[0]["position_side"] == "long"
    assert orders[0]["qty"] == 0.5
    assert orders[0]["custom_id"] == "entry_long_01"
    assert orders[0]["side"] == "buy"
