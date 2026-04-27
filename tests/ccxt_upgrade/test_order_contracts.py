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
