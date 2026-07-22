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
        (BinanceBot, {"positionSide": "BOTH"}, {}),
        (BybitBot, {"positionIdx": 0}, {}),
        (BitgetBot, {}, {"is_uta": False}),
        (HyperliquidBot, {}, {}),
        (GateIOBot, {}, {}),
        (KucoinBot, {}, {}),
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
        (BinanceBot, {"positionSide": "BOTH"}, {}),
        (BybitBot, {"positionIdx": 0}, {}),
        (BitgetBot, {}, {"is_uta": False}),
        (HyperliquidBot, {}, {}),
        (GateIOBot, {}, {}),
        (KucoinBot, {}, {}),
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


def test_symbol_market_metadata_epoch_changes_only_with_normalization_inputs():
    class Bot:
        price_steps = {"BTC/USDT:USDT": 0.1}
        qty_steps = {"BTC/USDT:USDT": 0.001}
        min_qtys = {"BTC/USDT:USDT": 0.001}
        min_costs = {"BTC/USDT:USDT": 5.0}
        c_mults = {"BTC/USDT:USDT": 1.0}
        markets_dict = {
            "BTC/USDT:USDT": {
                "active": True,
                "precision": {"price": 0.1, "amount": 0.001},
                "limits": {"amount": {"min": 0.001}},
                "contractSize": 1.0,
                "linear": True,
                "inverse": False,
                "type": "swap",
                "info": {"volatile_field": 1},
            }
        }

    bot = Bot()
    first = reconciler._order_churn_symbol_compatibility_epochs(
        bot, {"BTC/USDT:USDT"}
    )
    bot.markets_dict["BTC/USDT:USDT"]["info"]["volatile_field"] = 2
    assert reconciler._order_churn_symbol_compatibility_epochs(
        bot, {"BTC/USDT:USDT"}
    ) == first

    bot.qty_steps["BTC/USDT:USDT"] = 0.01
    assert reconciler._order_churn_symbol_compatibility_epochs(
        bot, {"BTC/USDT:USDT"}
    ) != first


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price_step", 0.01),
        ("qty_step", 0.01),
        ("min_qty", 0.01),
        ("min_cost", 10.0),
        ("contract_multiplier", 2.0),
        ("active", False),
        ("precision", {"price": 0.01, "amount": 0.001}),
        ("limits", {"amount": {"min": 0.01}}),
        ("contract_size", 2.0),
        ("linear", False),
        ("inverse", True),
        ("type", "future"),
    ],
)
def test_hourly_market_refresh_changes_reset_symbol_compatibility_epoch(field, value):
    symbol = "BTC/USDT:USDT"

    class Bot:
        price_steps = {symbol: 0.1}
        qty_steps = {symbol: 0.001}
        min_qtys = {symbol: 0.001}
        min_costs = {symbol: 5.0}
        c_mults = {symbol: 1.0}
        markets_dict = {
            symbol: {
                "active": True,
                "precision": {"price": 0.1, "amount": 0.001},
                "limits": {"amount": {"min": 0.001}},
                "contractSize": 1.0,
                "linear": True,
                "inverse": False,
                "type": "swap",
            }
        }

    bot = Bot()
    baseline = reconciler._order_churn_symbol_compatibility_epochs(bot, {symbol})
    if field == "price_step":
        bot.price_steps[symbol] = value
    elif field == "qty_step":
        bot.qty_steps[symbol] = value
    elif field == "min_qty":
        bot.min_qtys[symbol] = value
    elif field == "min_cost":
        bot.min_costs[symbol] = value
    elif field == "contract_multiplier":
        bot.c_mults[symbol] = value
    else:
        market_key = {"contract_size": "contractSize"}.get(field, field)
        bot.markets_dict[symbol][market_key] = value

    assert reconciler._order_churn_symbol_compatibility_epochs(bot, {symbol}) != baseline


def test_account_epoch_tracks_realized_and_global_inputs_not_scoped_runtime_policy():
    symbol = "BTC/USDT:USDT"

    class Bot:
        positions = {
            symbol: {
                "long": {
                    "size": 1.0,
                    "price": 100.0,
                    "unrealized_pnl": 9.0,
                    "liquidation_price": 50.0,
                },
                "short": {"size": 0.0, "price": 0.0},
            }
        }
        config = {"live": {"execution_delay_seconds": 2.0}, "bot": {}}
        PB_modes = {"long": {symbol: "normal"}, "short": {symbol: "normal"}}
        approved_coins = {"long": {symbol}, "short": {symbol}}
        ignored_coins = {"long": set(), "short": set()}
        approved_coins_minus_ignored_coins = {
            "long": {symbol},
            "short": {symbol},
        }
        active_symbols = [symbol]
        _config_hedge_mode = True
        hedge_mode = True
        _pnls_manager = None
        equity = 109.0
        available_margin = 80.0

        def __init__(self):
            self.raw_balance = 100.0
            self.snapped_balance = 100.0
            self.pnl_max = 2.0
            self.pnl_last = 1.0

        def get_hysteresis_snapped_balance(self):
            return self.snapped_balance

        def get_raw_balance(self):
            return self.raw_balance

        def _get_realized_pnl_cumsum_stats(self):
            return {"max": self.pnl_max, "last": self.pnl_last}

    bot = Bot()
    baseline = reconciler._order_churn_account_epoch(bot)
    bot.equity = 120.0
    bot.available_margin = 70.0
    bot.positions[symbol]["long"]["unrealized_pnl"] = 20.0
    bot.positions[symbol]["long"]["liquidation_price"] = 60.0
    assert reconciler._order_churn_account_epoch(bot) == baseline

    # Raw balance may be quote-valued collateral and therefore move with the
    # market even without a fill or transfer. Rust sizing uses the snapped
    # balance, while fills/PnL/positions provide phase-change evidence.
    bot.raw_balance = 101.0
    assert reconciler._order_churn_account_epoch(bot) == baseline
    bot.raw_balance = 100.0
    bot.snapped_balance = 101.0
    assert reconciler._order_churn_account_epoch(bot) != baseline
    bot.snapped_balance = 100.0

    bot.positions[symbol]["long"]["size"] = 1.1
    assert reconciler._order_churn_account_epoch(bot) != baseline
    bot.positions[symbol]["long"]["size"] = 1.0
    bot.pnl_last = 1.1
    assert reconciler._order_churn_account_epoch(bot) != baseline
    bot.pnl_last = 1.0
    bot.config["live"]["execution_delay_seconds"] = 3.0
    assert reconciler._order_churn_account_epoch(bot) != baseline
    bot.config["live"]["execution_delay_seconds"] = 2.0
    # Runtime forager/mode/list changes are symbol-scoped: resetting them
    # account-wide lets a rotating empty slot erase unrelated churn evidence.
    bot.active_symbols = [symbol, "ETH/USDT:USDT"]
    bot.PB_modes["long"][symbol] = "graceful_stop"
    bot.approved_coins["long"].clear()
    bot.ignored_coins["long"].add(symbol)
    bot.approved_coins_minus_ignored_coins["long"].clear()
    assert reconciler._order_churn_account_epoch(bot) == baseline

    bot._authoritative_surface_signatures = {"fills": ((1, "fill-a"),)}
    with_fill_signature = reconciler._order_churn_account_epoch(bot)
    bot._authoritative_surface_signatures["fills"] = ((1, "fill-b"),)
    assert reconciler._order_churn_account_epoch(bot) != with_fill_signature


@pytest.mark.parametrize(
    "mutation",
    [
        "mode",
        "approved",
        "ignored",
        "approved_minus_ignored",
        "active",
    ],
)
def test_runtime_policy_changes_reset_only_the_affected_symbol_epoch(mutation):
    btc = "BTC/USDT:USDT"
    eth = "ETH/USDT:USDT"

    class Bot:
        price_steps = {btc: 0.1, eth: 0.01}
        qty_steps = {btc: 0.001, eth: 0.01}
        min_qtys = {btc: 0.001, eth: 0.01}
        min_costs = {btc: 5.0, eth: 5.0}
        c_mults = {btc: 1.0, eth: 1.0}
        markets_dict = {btc: {"active": True}, eth: {"active": True}}
        PB_modes = {
            "long": {btc: "normal", eth: "normal"},
            "short": {btc: "normal", eth: "normal"},
        }
        approved_coins = {"long": {btc, eth}, "short": {btc, eth}}
        ignored_coins = {"long": set(), "short": set()}
        approved_coins_minus_ignored_coins = {
            "long": {btc, eth},
            "short": {btc, eth},
        }
        active_symbols = [btc, eth]

    bot = Bot()
    baseline = reconciler._order_churn_symbol_compatibility_epochs(bot, {btc, eth})
    if mutation == "mode":
        bot.PB_modes["long"][btc] = "graceful_stop"
    elif mutation == "approved":
        bot.approved_coins["long"].remove(btc)
    elif mutation == "ignored":
        bot.ignored_coins["long"].add(btc)
    elif mutation == "approved_minus_ignored":
        bot.approved_coins_minus_ignored_coins["long"].remove(btc)
    else:
        bot.active_symbols.remove(btc)

    changed = reconciler._order_churn_symbol_compatibility_epochs(bot, {btc, eth})
    assert changed[btc] != baseline[btc]
    assert changed[eth] == baseline[eth]
