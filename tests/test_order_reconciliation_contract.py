from __future__ import annotations

import passivbot_rust as pbr
import pytest

from exchanges.binance import BinanceBot
from exchanges.bitget import BitgetBot
from exchanges.bybit import BybitBot
from exchanges.ccxt_bot import CCXTBot
from exchanges.gateio import GateIOBot
from exchanges.hyperliquid import HyperliquidBot
from exchanges.kucoin import KucoinBot
from exchanges.okx import OKXBot
from exchanges.weex import WeexBot
from live import reconciler


def _client_id(pb_order_type: str) -> str:
    return f"pb-0x{pbr.order_type_snake_to_id(pb_order_type):04x}-test"


def test_hyperliquid_native_cloid_is_a_client_order_identity():
    client_id = _client_id("entry_grid_normal_long")

    assert reconciler.extract_order_custom_id({"info": {"cloid": client_id}}) == client_id


@pytest.mark.parametrize(
    ("bot_cls", "extra", "side", "expected_pside", "expected_close"),
    [
        (BinanceBot, {"info": {"ps": "LONG"}}, "buy", "long", False),
        (BinanceBot, {"info": {"ps": "LONG"}}, "sell", "long", True),
        (BybitBot, {"info": {"positionIdx": 2}}, "sell", "short", False),
        (BybitBot, {"info": {"positionIdx": 2}}, "buy", "short", True),
        (HyperliquidBot, {"reduceOnly": False}, "buy", "long", False),
        (HyperliquidBot, {"reduceOnly": True}, "sell", "long", True),
    ],
)
def test_one_way_and_hedge_action_tuples_have_deterministic_pside_and_close_effect(
    bot_cls, extra, side, expected_pside, expected_close
):
    bot = bot_cls.__new__(bot_cls)
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": side,
        "clientOrderId": _client_id(
            ("close_grid_" if expected_close else "entry_grid_normal_")
            + expected_pside
        ),
        **extra,
    }
    assert bot._get_position_side_for_order(order) == expected_pside
    assert bot._canonical_open_order_reduce_only(order) is expected_close


def test_gateio_accepts_native_is_reduce_only_response_field():
    bot = GateIOBot.__new__(GateIOBot)
    close = {
        "side": "sell",
        "info": {"side": "sell", "is_reduce_only": True},
    }

    assert bot._canonical_open_order_reduce_only(close) is True
    assert bot.determine_pos_side(close) == "long"


@pytest.mark.parametrize(
    "bot_cls",
    [BinanceBot, BitgetBot, BybitBot, KucoinBot, OKXBot],
)
def test_supported_hedge_orders_do_not_fabricate_pside_from_client_id(bot_cls):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = True
    bot.hedge_mode = True
    if bot_cls is BitgetBot:
        bot.is_uta = False
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "clientOrderId": _client_id("entry_grid_normal_long"),
        "info": {},
    }

    with pytest.raises(ValueError, match="missing"):
        bot._get_position_side_for_order(order)


def test_bitget_uta_websocket_order_uses_native_hold_side():
    bot = BitgetBot.__new__(BitgetBot)
    bot._config_hedge_mode = True
    bot.hedge_mode = True
    bot.is_uta = True
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "amount": 0.1,
        "info": {
            "holdMode": "hedge_mode",
            "holdSide": "long",
            "side": "sell",
            "tradeSide": "close",
        },
    }

    normalized = bot._normalize_order_update(order)

    assert normalized["side"] == "sell"
    assert normalized["position_side"] == "long"
    assert normalized["qty"] == 0.1


@pytest.mark.parametrize(
    "progress",
    [
        {"filled": 0.04, "remaining": 0.06},
        {"filled": 0.0, "remaining": 0.06},
    ],
)
def test_bitget_uta_websocket_partial_fill_requires_authoritative_refresh(progress):
    bot = BitgetBot.__new__(BitgetBot)
    bot._config_hedge_mode = True
    bot.hedge_mode = True
    bot.is_uta = True
    order = {
        "symbol": "BTC/USDT:USDT",
        "status": "open",
        "side": "buy",
        "amount": 0.1,
        **progress,
        "info": {"holdSide": "long", "side": "buy"},
    }

    normalized = bot._normalize_order_update(order)

    assert normalized["_pb_order_update_requires_authoritative_refresh"] is True


def test_bitget_uta_websocket_unfilled_open_does_not_force_refresh():
    bot = BitgetBot.__new__(BitgetBot)
    bot._config_hedge_mode = True
    bot.hedge_mode = True
    bot.is_uta = True
    order = {
        "symbol": "BTC/USDT:USDT",
        "status": "open",
        "side": "buy",
        "amount": 0.1,
        "filled": 0.0,
        "remaining": 0.1,
        "info": {"holdSide": "long", "side": "buy"},
    }

    normalized = bot._normalize_order_update(order)

    assert "_pb_order_update_requires_authoritative_refresh" not in normalized


def test_bitget_uta_websocket_order_rejects_conflicting_native_psides():
    bot = BitgetBot.__new__(BitgetBot)
    bot.hedge_mode = True
    bot.is_uta = True

    with pytest.raises(ValueError, match="authoritative position-side"):
        bot._get_position_side_for_order(
            {"info": {"posSide": "short", "holdSide": "long"}}
        )


@pytest.mark.parametrize("bot_cls", [BinanceBot, KucoinBot])
def test_sparse_websocket_order_update_uses_passivbot_client_pside(bot_cls):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = True
    bot.hedge_mode = True
    bot.get_exchange_time = lambda: 1_000
    client_id = _client_id("entry_grid_normal_long")
    bot.orders_emitted_to_exchange = [
        {
            "timestamp": 900,
            "exchange_id": "",
            "canonical_custom_id": bot._canonical_passivbot_custom_id(client_id),
        }
    ]
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "amount": 0.1,
        "clientOrderId": client_id,
        "info": {},
    }

    normalized = bot._normalize_order_update(order)

    assert normalized["position_side"] == "long"
    assert normalized["qty"] == 0.1
    assert normalized["_pb_order_update_requires_authoritative_refresh"] is True


@pytest.mark.parametrize("bot_cls", [BinanceBot, KucoinBot])
def test_sparse_non_passivbot_websocket_order_update_remains_rejected(bot_cls):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = True
    bot.hedge_mode = True
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "amount": 0.1,
        "clientOrderId": "user-order",
        "info": {},
    }

    with pytest.raises(ValueError, match="missing"):
        bot._normalize_order_update(order)


@pytest.mark.parametrize("bot_cls", [BinanceBot, KucoinBot])
def test_sparse_foreign_passivbot_websocket_order_update_remains_rejected(bot_cls):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = True
    bot.hedge_mode = True
    bot.get_exchange_time = lambda: 1_000
    bot.orders_emitted_to_exchange = [
        {
            "timestamp": 900,
            "exchange_id": "",
            "canonical_custom_id": bot._canonical_passivbot_custom_id(
                _client_id("entry_grid_normal_long") + "-ours"
            ),
        }
    ]
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "amount": 0.1,
        "clientOrderId": _client_id("entry_grid_normal_long") + "-foreign",
        "info": {},
    }

    with pytest.raises(ValueError, match="missing"):
        bot._normalize_order_update(order)


@pytest.mark.parametrize("bot_cls", [BinanceBot, KucoinBot])
def test_sparse_one_way_websocket_order_update_does_not_use_marker_recovery(bot_cls):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = False
    bot.hedge_mode = False
    bot.get_exchange_time = lambda: 1_000
    client_id = _client_id("entry_grid_normal_long")
    bot.orders_emitted_to_exchange = [
        {
            "timestamp": 900,
            "exchange_id": "",
            "canonical_custom_id": bot._canonical_passivbot_custom_id(client_id),
        }
    ]
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "amount": 0.1,
        "clientOrderId": client_id,
        "info": {},
    }

    with pytest.raises(ValueError):
        bot._normalize_order_update(order)


@pytest.mark.parametrize("bot_cls", [BinanceBot, KucoinBot])
def test_sparse_ws_recovery_uses_actual_exchange_hedge_mode(bot_cls):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = False
    bot.hedge_mode = True
    bot.get_exchange_time = lambda: 1_000
    client_id = _client_id("entry_grid_normal_long")
    bot.orders_emitted_to_exchange = [
        {
            "timestamp": 900,
            "exchange_id": "",
            "canonical_custom_id": bot._canonical_passivbot_custom_id(client_id),
        }
    ]
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "amount": 0.1,
        "clientOrderId": client_id,
        "info": {},
    }

    normalized = bot._normalize_order_update(order)

    assert normalized["position_side"] == "long"
    assert normalized["_pb_order_update_requires_authoritative_refresh"] is True


@pytest.mark.parametrize(
    ("bot_cls", "info"),
    [
        (BinanceBot, {"ps": "BOTH"}),
        (KucoinBot, {"positionSide": "BOTH"}),
    ],
)
def test_sparse_ws_recovery_rejects_explicit_native_one_way_metadata(bot_cls, info):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = True
    bot.hedge_mode = True
    bot.get_exchange_time = lambda: 1_000
    client_id = _client_id("entry_grid_normal_long")
    bot.orders_emitted_to_exchange = [
        {
            "timestamp": 900,
            "exchange_id": "",
            "canonical_custom_id": bot._canonical_passivbot_custom_id(client_id),
        }
    ]
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "amount": 0.1,
        "clientOrderId": client_id,
        "info": info,
    }

    with pytest.raises(ValueError):
        bot._normalize_order_update(order)


@pytest.mark.parametrize("bot_cls", [BinanceBot, KucoinBot])
def test_sparse_ws_recovery_rejects_conflicting_emitted_identities(bot_cls):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = True
    bot.hedge_mode = True
    bot.get_exchange_time = lambda: 1_000
    owned_client_id = _client_id("entry_grid_normal_long") + "-ours"
    foreign_client_id = _client_id("entry_grid_normal_short") + "-foreign"
    bot.orders_emitted_to_exchange = [
        {
            "timestamp": 900,
            "exchange_id": "owned-exchange-id",
            "canonical_custom_id": bot._canonical_passivbot_custom_id(
                owned_client_id
            ),
        }
    ]
    order = {
        "id": "owned-exchange-id",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "amount": 0.1,
        "clientOrderId": foreign_client_id,
        "info": {},
    }

    with pytest.raises(ValueError, match="missing"):
        bot._normalize_order_update(order)


@pytest.mark.parametrize("bot_cls", [BinanceBot, KucoinBot])
@pytest.mark.parametrize(
    ("top_level", "raw_info"),
    [
        (
            {"id": "owned-exchange-id"},
            {"orderId": "foreign-exchange-id"},
        ),
        (
            {"clientOrderId": _client_id("entry_grid_normal_long") + "-ours"},
            {"clientOid": _client_id("entry_grid_normal_short") + "-foreign"},
        ),
    ],
)
def test_sparse_ws_recovery_rejects_conflicting_duplicate_identity_aliases(
    bot_cls, top_level, raw_info
):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = True
    bot.hedge_mode = True
    bot.get_exchange_time = lambda: 1_000
    owned_client_id = _client_id("entry_grid_normal_long") + "-ours"
    bot.orders_emitted_to_exchange = [
        {
            "timestamp": 900,
            "exchange_id": "owned-exchange-id",
            "canonical_custom_id": bot._canonical_passivbot_custom_id(
                owned_client_id
            ),
        }
    ]
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "amount": 0.1,
        "info": raw_info,
        **top_level,
    }

    with pytest.raises(ValueError, match="missing"):
        bot._normalize_order_update(order)


@pytest.mark.parametrize(
    ("side", "position_side", "reported_reduce_only", "expected_close"),
    [
        ("buy", "LONG", False, False),
        ("sell", "SHORT", False, False),
        # WEEX V3 COMBINED has been observed returning reduceOnly=false for
        # ordinary closes even though side + positionSide closes the position.
        ("sell", "LONG", False, True),
        ("buy", "SHORT", False, True),
        # The response field is not part of the V3 placement contract, so the
        # authoritative action tuple also wins when that literal is true.
        ("buy", "LONG", True, False),
    ],
)
def test_weex_uses_v3_action_tuple_not_response_reduce_only_literal(
    side, position_side, reported_reduce_only, expected_close
):
    bot = WeexBot.__new__(WeexBot)
    order = {
        "side": side,
        "reduceOnly": reported_reduce_only,
        "info": {
            "side": side.upper(),
            "positionSide": position_side,
            "reduceOnly": reported_reduce_only,
        },
    }
    assert bot._canonical_open_order_reduce_only(order) is expected_close


@pytest.mark.parametrize(
    ("side", "position_side", "expected_close"),
    [
        ("buy", "LONG", False),
        ("sell", "SHORT", False),
        ("sell", "LONG", True),
        ("buy", "SHORT", True),
    ],
)
def test_kucoin_hedge_mode_uses_side_position_tuple_without_reduce_only(
    side, position_side, expected_close
):
    bot = KucoinBot.__new__(KucoinBot)
    bot.hedge_mode = True
    order = {
        "side": side,
        "info": {
            "side": side,
            "positionSide": position_side,
        },
    }

    assert bot._get_position_side_for_order(order) == position_side.lower()
    assert bot._canonical_open_order_reduce_only(order) is expected_close


def test_bitget_uta_uses_action_tuple_even_when_literal_reduce_only_is_false():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = True
    close = {
        "side": "sell",
        "reduceOnly": False,
        "info": {"posSide": "LONG", "reduceOnly": "NO"},
    }
    assert bot._canonical_open_order_reduce_only(close) is True


def test_okx_hedge_mode_uses_action_tuple_when_ccxt_reduce_only_is_false():
    bot = OKXBot.__new__(OKXBot)
    close = {
        "side": "sell",
        "reduceOnly": False,
        "info": {"posSide": "long", "side": "sell", "reduceOnly": "false"},
    }
    entry = {
        "side": "buy",
        "reduceOnly": False,
        "info": {"posSide": "long", "side": "buy", "reduceOnly": "false"},
    }

    assert bot._canonical_open_order_reduce_only(close) is True
    assert bot._canonical_open_order_reduce_only(entry) is False


def test_ccxt_close_only_normalization_does_not_trust_parser_default_over_raw_info():
    bot = HyperliquidBot.__new__(HyperliquidBot)

    assert (
        bot._strict_order_reduce_only_response(
            {"reduceOnly": False, "info": {"someNativeField": "value"}}
        )
        is None
    )
    assert (
        bot._strict_order_reduce_only_response(
            {"reduceOnly": False, "info": {"reduceOnly": "true"}}
        )
        is True
    )


@pytest.mark.parametrize(
    "raw_position_idx", [True, False, "1.5", 1.5, float("nan"), 3]
)
def test_bybit_rejects_noncanonical_position_idx(raw_position_idx):
    bot = BybitBot.__new__(BybitBot)
    order = {
        "side": "buy",
        "reduceOnly": False,
        "info": {"positionIdx": raw_position_idx, "reduceOnly": False},
    }

    with pytest.raises(ValueError, match="positionIdx"):
        bot._get_position_side_for_order(order)
    with pytest.raises(ValueError, match="positionIdx"):
        bot._canonical_open_order_reduce_only(order)


@pytest.mark.parametrize(
    ("bot_cls", "info", "extra_attrs"),
    [
        (BinanceBot, {"positionSide": "BOTH"}, {"hedge_mode": False}),
        (BybitBot, {"positionIdx": 0}, {}),
        (BitgetBot, {}, {"is_uta": False, "hedge_mode": False}),
        (HyperliquidBot, {}, {}),
        (GateIOBot, {}, {}),
        (KucoinBot, {}, {"hedge_mode": False}),
        (OKXBot, {"posSide": "net"}, {}),
    ],
)
@pytest.mark.parametrize(
    ("side", "reduce_only", "expected_pside"),
    [
        ("buy", False, "long"),
        ("sell", False, "short"),
        ("buy", True, "short"),
        ("sell", True, "long"),
    ],
)
def test_effective_one_way_orders_cover_all_side_close_only_tuples(
    bot_cls, info, extra_attrs, side, reduce_only, expected_pside
):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = False
    bot.hedge_mode = getattr(bot, "hedge_mode", True)
    for key, value in extra_attrs.items():
        setattr(bot, key, value)
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": side,
        "reduceOnly": reduce_only,
        "info": {**info, "reduceOnly": reduce_only},
    }

    assert bot._get_position_side_for_order(order) == expected_pside
    assert bot._canonical_open_order_reduce_only(order) is reduce_only


@pytest.mark.parametrize(
    ("bot_cls", "info", "extra_attrs"),
    [
        (BinanceBot, {"positionSide": "BOTH"}, {"hedge_mode": False}),
        (BybitBot, {"positionIdx": 0}, {}),
        (BitgetBot, {}, {"is_uta": False, "hedge_mode": False}),
        (HyperliquidBot, {}, {}),
        (GateIOBot, {}, {}),
        (KucoinBot, {}, {"hedge_mode": False}),
        (OKXBot, {"posSide": "net"}, {}),
    ],
)
def test_one_way_metadata_must_agree_with_action_tuple(bot_cls, info, extra_attrs):
    bot = bot_cls.__new__(bot_cls)
    bot._config_hedge_mode = False
    bot.hedge_mode = getattr(bot, "hedge_mode", True)
    for key, value in extra_attrs.items():
        setattr(bot, key, value)
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "reduceOnly": False,
        "clientOrderId": _client_id("entry_grid_normal_short"),
        "info": {**info, "reduceOnly": False},
    }

    with pytest.raises(ValueError, match="contradicts"):
        bot._get_position_side_for_order(order)


def test_one_way_pside_does_not_trust_arbitrary_hex_prefixed_client_id():
    bot = CCXTBot.__new__(CCXTBot)
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "reduceOnly": False,
        "clientOrderId": "0007-user-created",
        "info": {"reduceOnly": False},
    }

    assert bot._normalize_one_way_position_side(order) == "short"


@pytest.mark.parametrize("bot_cls", [BinanceBot, KucoinBot, OKXBot])
def test_open_order_position_side_never_defaults_from_current_position(bot_cls):
    bot = bot_cls.__new__(bot_cls)
    bot.positions = {
        "BTC/USDT:USDT": {
            "long": {"size": 1.0},
            "short": {"size": 0.0},
        }
    }
    with pytest.raises((ValueError, Exception)):
        bot._get_position_side_for_order(
            {"symbol": "BTC/USDT:USDT", "side": "sell", "info": {}}
        )


def test_binance_one_way_open_order_normalizer_uses_action_tuple():
    bot = BinanceBot.__new__(BinanceBot)
    bot._config_hedge_mode = False
    bot.hedge_mode = False
    bot.markets_dict = {}
    bot._record_live_margin_mode_from_payload = lambda _order: None
    order = {
        "id": "1",
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "amount": 1.0,
        "timestamp": 1,
        "reduceOnly": False,
        "info": {
            "symbol": "BTCUSDT",
            "positionSide": "BOTH",
            "reduceOnly": False,
        },
    }

    [normalized] = bot._normalize_open_orders([order])

    assert normalized["position_side"] == "short"


def test_binance_exchange_hedge_order_ignores_strategy_one_way_setting():
    bot = BinanceBot.__new__(BinanceBot)
    bot._config_hedge_mode = False
    bot.hedge_mode = True
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "info": {"positionSide": "LONG"},
    }

    assert bot._get_position_side_for_order(order) == "long"
    assert bot._canonical_open_order_reduce_only(order) is True


def test_bitget_one_way_open_order_normalizer_uses_action_tuple():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = False
    bot._config_hedge_mode = False
    bot.hedge_mode = False
    bot._record_live_margin_mode_from_payload = lambda _order: None
    order = {
        "id": "1",
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "amount": 1.0,
        "timestamp": 1,
        "clientOrderId": "",
        "reduceOnly": False,
        "info": {"side": "sell", "posSide": "net", "reduceOnly": False},
    }

    [normalized] = bot._normalize_open_orders([order])

    assert normalized["position_side"] == "short"
    assert normalized["side"] == "sell"


def test_bitget_exchange_hedge_order_ignores_strategy_one_way_setting():
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = False
    bot._config_hedge_mode = False
    bot.hedge_mode = True
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "info": {"side": "sell", "posSide": "long", "tradeSide": "close"},
    }

    assert bot._get_position_side_for_order(order) == "long"
    assert bot._canonical_open_order_reduce_only(order) is True


@pytest.mark.parametrize(
    ("side", "pos_side", "expected_close"),
    [
        ("buy", "long", False),
        ("sell", "long", True),
        ("sell", "short", False),
        ("buy", "short", True),
    ],
)
def test_bitget_uta_one_way_open_order_normalizer_uses_action_tuple_without_recursion(
    side, pos_side, expected_close
):
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = True
    bot._config_hedge_mode = False
    bot.hedge_mode = False
    bot._record_live_margin_mode_from_payload = lambda _order: None
    order = {
        "id": "1",
        "symbol": "BTC/USDT:USDT",
        "side": side,
        "amount": 1.0,
        "timestamp": 1,
        "clientOrderId": "",
        # Bitget UTA may report the literal as false for an action which
        # closes the explicit posSide.  The action tuple is authoritative.
        "reduceOnly": False,
        "info": {"side": side, "posSide": pos_side, "reduceOnly": "NO"},
    }

    [normalized] = bot._normalize_open_orders([order])

    assert normalized["position_side"] == pos_side
    assert normalized["side"] == side
    assert bot._canonical_open_order_reduce_only(normalized) is expected_close


@pytest.mark.parametrize(
    ("side", "trade_side", "expected_pside", "expected_close"),
    [
        ("buy", "open", "long", False),
        ("sell", "open", "short", False),
        ("sell", "close", "long", True),
        ("buy", "close", "short", True),
    ],
)
def test_bitget_classic_one_way_uses_trade_side_without_reduce_only(
    side, trade_side, expected_pside, expected_close
):
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = False
    bot._config_hedge_mode = False
    bot.hedge_mode = False
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": side,
        "info": {"side": side, "tradeSide": trade_side},
    }

    assert bot._get_position_side_for_order(order) == expected_pside
    assert bot._canonical_open_order_reduce_only(order) is expected_close


class _SnapshotBot:
    def __init__(self, orders):
        self.active_symbols = []
        self.open_orders = {"BTC/USDT:USDT": orders}
        self.positions = {}
        self.dirty = []

    def _canonical_open_order_reduce_only(self, order):
        return reconciler.extract_order_reduce_only(order)

    def _mark_account_critical_state_dirty(self, **kwargs):
        self.dirty.append(kwargs)


def test_actual_snapshot_uses_remaining_quantity_and_includes_open_order_only_symbols():
    order = {
        "id": "1",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "amount": 2.0,
        "filled": 0.5,
        "remaining": 1.5,
        "price": 100.0,
        "reduceOnly": False,
        "type": "limit",
        "clientOrderId": _client_id("entry_grid_normal_long"),
    }
    bot = _SnapshotBot([order])
    snapshot = reconciler.snapshot_actual_orders(bot)
    assert snapshot["BTC/USDT:USDT"][0]["qty"] == 1.5
    assert bot.dirty == []


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda order: order.pop("id"), id="missing-id"),
        pytest.param(
            lambda order: order.update(symbol="ETH/USDT:USDT"),
            id="symbol-bucket-mismatch",
        ),
        pytest.param(lambda order: order.pop("side"), id="missing-side"),
        pytest.param(lambda order: order.update(side="hold"), id="invalid-side"),
        pytest.param(
            lambda order: order.pop("position_side"), id="missing-position-side"
        ),
        pytest.param(
            lambda order: order.update(position_side="both"),
            id="invalid-position-side",
        ),
        pytest.param(lambda order: order.pop("price"), id="missing-price"),
        pytest.param(lambda order: order.update(price=0.0), id="zero-price"),
        pytest.param(
            lambda order: order.update(price=float("nan")), id="nonfinite-price"
        ),
        pytest.param(lambda order: order.pop("remaining"), id="missing-remaining"),
        pytest.param(
            lambda order: order.update(remaining=float("nan")),
            id="nonfinite-remaining",
        ),
        pytest.param(
            lambda order: order.pop("reduceOnly"), id="missing-close-only"
        ),
    ],
)
def test_malformed_open_order_identity_makes_account_surface_unavailable(mutation):
    order = {
        "id": "1",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "remaining": 1.5,
        "price": 100.0,
        "reduceOnly": False,
        "type": "limit",
        "clientOrderId": _client_id("entry_grid_normal_long"),
    }
    mutation(order)
    bot = _SnapshotBot([order])

    assert reconciler.snapshot_actual_orders(bot)["BTC/USDT:USDT"] == []
    assert bot._malformed_actual_order_symbols == {"BTC/USDT:USDT"}
    assert bot.dirty[0]["reason"] == "malformed_open_order_snapshot"


def test_scoped_protective_snapshot_still_validates_all_account_open_orders():
    valid = {
        "id": "btc-1",
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "position_side": "long",
        "remaining": 1.0,
        "price": 101.0,
        "reduceOnly": True,
        "type": "limit",
        "clientOrderId": _client_id("close_panic_long"),
    }
    malformed_elsewhere = {
        "symbol": "ETH/USDT:USDT",
        "side": "buy",
        "position_side": "short",
        "remaining": 1.0,
        "price": 99.0,
        "reduceOnly": True,
        "type": "limit",
        "clientOrderId": _client_id("close_panic_short"),
    }
    bot = _SnapshotBot([valid])
    bot.open_orders["ETH/USDT:USDT"] = [malformed_elsewhere]

    snapshot = reconciler.snapshot_actual_orders(
        bot,
        ["BTC/USDT:USDT"],
        psides_by_symbol={"BTC/USDT:USDT": {"long"}},
    )

    assert list(snapshot) == ["BTC/USDT:USDT"]
    assert snapshot["BTC/USDT:USDT"][0]["id"] == "btc-1"
    assert bot._malformed_actual_order_symbols == {"ETH/USDT:USDT"}
    assert bot.dirty[0]["symbols"] == {"ETH/USDT:USDT"}


def test_malformed_unmanaged_order_still_blocks_account_open_order_surface():
    symbol = "BTC/USDT:USDT"
    malformed_manual_order = {
        # Missing exchange order ID remains account-critical even though the
        # long side is otherwise outside bot-managed reconciliation scope.
        "symbol": symbol,
        "side": "buy",
        "position_side": "long",
        "remaining": 1.0,
        "price": 100.0,
        "reduceOnly": False,
        "type": "limit",
    }
    bot = _SnapshotBot([malformed_manual_order])
    bot.PB_modes = {"long": {symbol: "manual"}, "short": {symbol: "normal"}}

    assert reconciler.snapshot_actual_orders(bot)[symbol] == []
    assert bot._malformed_actual_order_symbols == {symbol}
    assert bot.dirty[0]["reason"] == "malformed_open_order_snapshot"


@pytest.mark.parametrize(
    "open_orders",
    [
        {"": [{"id": "orphan"}]},
        {None: [{"id": "orphan"}]},
        {123: [{"id": "orphan"}]},
        {"BTC/USDT:USDT": {"id": "not-a-list"}},
        ["not-a-symbol-map"],
    ],
)
def test_malformed_open_order_container_or_bucket_blocks_account_surface(open_orders):
    bot = _SnapshotBot([])
    bot.open_orders = open_orders

    reconciler.snapshot_actual_orders(bot)

    assert bot._malformed_actual_order_symbols
    assert bot.dirty[0]["reason"] == "malformed_open_order_snapshot"


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ({"amount": 2.0, "filled": 0.5, "remaining": 1.5}, 1.5),
        ({"amount": 2.0, "filled": 0.5}, 1.5),
        ({"qty": 1.5}, 1.5),
        ({"amount": 2.0}, None),
        ({"amount": 2.0, "filled": 0.5, "remaining": 1.4}, None),
        ({"amount": 2.0, "filled": 2.5, "remaining": 0.0}, None),
    ],
)
def test_remaining_open_quantity_requires_consistent_authoritative_fields(order, expected):
    assert reconciler.extract_order_remaining_qty(order) == expected


def test_unsupported_generic_connector_uses_legacy_open_order_identity():
    order = {
        "id": "generic-1",
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "position_side": "long",
        "qty": 0.25,
        "price": 101.0,
        # Generic fallback preserves its legacy side/pside reconciliation and
        # does not enable the audited exact-type contradiction guard.
        "clientOrderId": _client_id("entry_grid_normal_long"),
    }
    bot = _SnapshotBot([order])
    bot._order_churn_gate_enabled_for_connector = False

    [normalized] = reconciler.snapshot_actual_orders(bot)["BTC/USDT:USDT"]

    assert normalized["qty"] == 0.25
    assert normalized["reduce_only"] is True
    assert normalized["type"] == "unknown"
    assert normalized["pb_order_type"] == "entry_grid_normal_long"
    assert bot.dirty == []


def test_unknown_order_type_is_managed_but_unknown_close_semantics_fail_closed():
    user_order = {
        "id": "manual-user-order",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "remaining": 1.0,
        "price": 90.0,
        "reduceOnly": False,
        "type": "stop_market",
    }
    bot = _SnapshotBot([user_order])
    [normalized] = reconciler.snapshot_actual_orders(bot)["BTC/USDT:USDT"]
    assert normalized["pb_order_type"] == "unknown"
    assert normalized["type"] == "unknown"
    to_cancel, to_create = reconciler.reconcile_symbol_orders(
        bot,
        "BTC/USDT:USDT",
        [normalized],
        [],
        (
            "symbol",
            "side",
            "position_side",
            "reduce_only",
            "type",
            "pb_order_type",
            "qty",
            "price",
        ),
        apply_mode_filters=False,
    )
    assert to_cancel == [normalized]
    assert to_create == []

    unknown_semantics = dict(user_order)
    unknown_semantics.pop("reduceOnly")
    malformed_bot = _SnapshotBot([unknown_semantics])
    assert reconciler.snapshot_actual_orders(malformed_bot)["BTC/USDT:USDT"] == []
    assert malformed_bot._malformed_actual_order_symbols == {"BTC/USDT:USDT"}


def test_arbitrary_hex_prefixed_client_id_is_not_trusted_as_pb_order_type():
    user_order = {
        "id": "user-order",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "remaining": 1.0,
        "price": 90.0,
        "reduceOnly": False,
        "type": "limit",
        # The legacy decoder accepts a leading four-hex token for historical
        # fill compatibility. Resting-order semantics require the explicit PB
        # marker so an unrelated user ID cannot fabricate a cohort type.
        "clientOrderId": "0007-user-created",
    }
    bot = _SnapshotBot([user_order])

    [normalized] = reconciler.snapshot_actual_orders(bot)["BTC/USDT:USDT"]

    assert normalized["pb_order_type"] == "unknown"
    assert bot.dirty == []


def test_mode_scope_contract_is_management_not_order_ownership():
    symbol = "BTC/USDT:USDT"
    entry = {
        "symbol": symbol,
        "position_side": "long",
        "reduce_only": False,
    }
    close = {
        "symbol": symbol,
        "position_side": "long",
        "reduce_only": True,
    }

    class Bot:
        PB_modes = {"long": {symbol: "manual"}, "short": {}}

    assert reconciler.apply_mode_filters(Bot(), symbol, [entry, close], [entry, close]) == (
        [],
        [],
    )
    Bot.PB_modes["long"][symbol] = "tp_only"
    assert reconciler.apply_mode_filters(Bot(), symbol, [entry, close], [entry, close]) == (
        [close],
        [close],
    )
    Bot.PB_modes["long"][symbol] = "tp_only_with_active_entry_cancellation"
    assert reconciler.apply_mode_filters(Bot(), symbol, [entry, close], [entry, close]) == (
        [entry, close],
        [close],
    )
    for fully_managed_mode in ("normal", "panic", "graceful_stop"):
        Bot.PB_modes["long"][symbol] = fully_managed_mode
        assert reconciler.apply_mode_filters(
            Bot(), symbol, [entry, close], [entry, close]
        ) == ([entry, close], [entry, close])


def test_manual_stop_preserves_only_forager_ema_entry_cancellation():
    symbol = "BTC/USDT:USDT"
    entry = {
        "id": "forager-entry",
        "symbol": symbol,
        "position_side": "long",
        "reduce_only": False,
    }
    close = {
        "id": "forager-close",
        "symbol": symbol,
        "position_side": "long",
        "reduce_only": True,
    }
    operator_entry = {
        "id": "operator-entry",
        "symbol": symbol,
        "position_side": "long",
        "reduce_only": False,
    }

    class Bot:
        PB_modes = {"long": {symbol: "manual"}, "short": {}}
        _orchestrator_ema_entry_cancellation_order_keys = {
            (symbol, "long", "exchange_id", "forager-entry")
        }

    assert reconciler.apply_mode_filters(
        Bot(), symbol, [entry, operator_entry, close], [entry, close]
    ) == ([entry], [])

    entry_with_both_ids = {
        **entry,
        "id": "exchange-id-arrived-later",
        "client_order_id": "forager-entry-client",
    }
    Bot._orchestrator_ema_entry_cancellation_order_keys = {
        (symbol, "long", "client_id", "forager-entry-client")
    }
    assert reconciler.apply_mode_filters(
        Bot(), symbol, [entry_with_both_ids, operator_entry, close], []
    ) == ([entry_with_both_ids], [])

    Bot._orchestrator_ema_entry_cancellation_order_keys = {
        (symbol, "short", "exchange_id", "forager-entry")
    }
    assert reconciler.apply_mode_filters(
        Bot(), symbol, [entry, close], [entry, close]
    ) == ([], [])


def _normalized_order(price: float) -> dict:
    return {
        "symbol": "BTC/USDT:USDT",
        "position_side": "long",
        "side": "buy",
        "reduce_only": False,
        "type": "limit",
        "pb_order_type": "entry_grid_normal_long",
        "qty": 1.0,
        "price": price,
    }


def test_tolerance_reconciliation_maximizes_one_to_one_matches():
    class Bot:
        @staticmethod
        def live_value(key):
            assert key == "order_match_tolerance_pct"
            return 0.0002

    # The first current order can match either actual; the second can match only
    # 100.02. A greedy first-match pass can preserve only one, while the
    # deterministic cohort pass preserves both.
    actual = [_normalized_order(100.02), _normalized_order(100.0)]
    current = [_normalized_order(100.01), _normalized_order(100.03)]

    to_cancel, to_create, skipped = reconciler.apply_order_match_tolerance(
        Bot(), actual, current
    )

    assert skipped == 2
    assert to_cancel == []
    assert to_create == []


def test_tolerance_reconciliation_maps_sorted_matches_back_to_source_orders():
    class Bot:
        @staticmethod
        def live_value(key):
            assert key == "order_match_tolerance_pct"
            return 0.0002

    unmatched_actual = _normalized_order(200.0)
    matching_actual = _normalized_order(100.0)
    unmatched_current = _normalized_order(300.0)
    matching_current = _normalized_order(100.01)

    to_cancel, to_create, skipped = reconciler.apply_order_match_tolerance(
        Bot(),
        [unmatched_actual, matching_actual],
        [unmatched_current, matching_current],
    )

    assert skipped == 1
    assert to_cancel == [unmatched_actual]
    assert to_create == [unmatched_current]


def test_churn_evidence_never_preserves_a_stale_actual_order():
    class Bot:
        @staticmethod
        def live_value(key):
            assert key == "order_match_tolerance_pct"
            return 0.0002

    actual = _normalized_order(100.0)
    current = {
        **_normalized_order(100.1),
        "_churn_evidence": True,
        "_churn_reason": "continuous_price_drift",
    }

    to_cancel, to_create, skipped = reconciler.apply_order_match_tolerance(
        Bot(), [actual], [current]
    )

    assert skipped == 0
    assert to_cancel == [actual]
    assert to_create == [current]


def test_time_in_force_and_post_only_do_not_change_resting_order_identity():
    actual = {
        **_normalized_order(100.0),
        "time_in_force": "GTC",
        "post_only": False,
    }
    current = {
        **_normalized_order(100.0),
        "time_in_force": "PO",
        "post_only": True,
    }
    keys = (
        "symbol",
        "side",
        "position_side",
        "reduce_only",
        "type",
        "pb_order_type",
        "qty",
        "price",
    )

    to_cancel, to_create = reconciler.reconcile_symbol_orders(
        object(),
        "BTC/USDT:USDT",
        [actual],
        [current],
        keys,
        apply_mode_filters=False,
    )

    assert to_cancel == []
    assert to_create == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("pb_order_type", "entry_ema_anchor_long"),
        ("type", "market"),
        ("reduce_only", True),
    ],
)
def test_semantic_identity_mismatches_require_reconciliation(field, replacement):
    actual = _normalized_order(100.0)
    current = _normalized_order(100.0)
    current[field] = replacement
    keys = (
        "symbol",
        "side",
        "position_side",
        "reduce_only",
        "type",
        "pb_order_type",
        "qty",
        "price",
    )

    to_cancel, to_create = reconciler.reconcile_symbol_orders(
        object(),
        "BTC/USDT:USDT",
        [actual],
        [current],
        keys,
        apply_mode_filters=False,
    )

    assert to_cancel == [actual]
    assert to_create == [current]


def test_unsupported_generic_connector_preserves_legacy_tolerance_matching():
    class Bot:
        _order_churn_gate_enabled_for_connector = False

        @staticmethod
        def live_value(key):
            assert key == "order_match_tolerance_pct"
            return 0.0002

    actual = {
        **_normalized_order(100.0),
        "pb_order_type": "unknown",
        "type": "unknown",
    }
    current = _normalized_order(100.01)

    to_cancel, to_create, skipped = reconciler.apply_order_match_tolerance(
        Bot(), [actual], [current]
    )

    assert skipped == 1
    assert to_cancel == []
    assert to_create == []


def test_rust_risk_active_pairs_cover_risk_orders_and_loss_gate_blocks():
    idx_to_symbol = {
        0: "BTC/USDT:USDT",
        1: "ETH/USDT:USDT",
        2: "SOL/USDT:USDT",
    }
    out = {
        "orders": [
            {
                "symbol_idx": 0,
                "pside": "long",
                "execution_priority": "ordinary",
            },
            {
                "symbol_idx": 1,
                "pside": "short",
                "execution_priority": "risk_critical",
            },
        ],
        "diagnostics": {
            "loss_gate_blocks": [
                {"symbol_idx": 2, "pside": "long"},
                {"symbol_idx": 1, "pside": "short"},
            ]
        },
    }

    assert reconciler.order_churn_risk_active_pairs_from_rust_output(
        out, idx_to_symbol
    ) == (("ETH/USDT:USDT", "short"), ("SOL/USDT:USDT", "long"))
