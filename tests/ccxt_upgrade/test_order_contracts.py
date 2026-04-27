from types import SimpleNamespace

import pytest

from ccxt_contracts import build_contract_bot
from exchanges.bitget import BitgetBot
from exchanges.binance import BinanceBot
from exchanges.bybit import BybitBot
from exchanges.defx import DefxBot
from exchanges.gateio import GateIOBot
from exchanges.kucoin import KucoinBot
from exchanges.okx import OKXBot
from passivbot import custom_id_has_explicit_passivbot_marker, custom_id_to_snake
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

    async def update_pnls():
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


def test_bybit_position_side_uses_determine_pos_side_ccxt_contract():
    bot = BybitBot.__new__(BybitBot)
    assert bot._get_position_side_for_order({"info": {"positionIdx": 2}}) == "short"
    assert bot._get_position_side_for_order({"info": {"positionSide": "LONG"}}) == "long"


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
