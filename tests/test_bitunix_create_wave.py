from dataclasses import replace
from types import SimpleNamespace

import pytest

from live.bitunix_create_wave import close_is_passive, select_ordinary_close_wave
from live.market_snapshot import MarketSnapshot


BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
NOW = 100_000
AGE = 2_000


def quote(symbol):
    return MarketSnapshot(symbol, 98, 101, 100, NOW - 3, "fetch_tickers", NOW - 10)


def position(symbol, pside, qty="10", *, id_="position-a"):
    return {
        "symbol": symbol,
        "position_side": pside,
        "info": {"positionId": id_, "qty": qty},
    }


def close(symbol, pside, *, id_="client-id", qty=4.0, price=None, **changes):
    return {
        "symbol": symbol,
        "position_side": pside,
        "side": "sell" if pside == "long" else "buy",
        "reduce_only": True,
        "type": "limit",
        "execution_priority": "ordinary",
        "pb_order_type": f"close_grid_{pside}",
        "custom_id": id_,
        "qty": qty,
        "price": price if price is not None else (101.0 if pside == "long" else 98.0),
        **changes,
    }


def venue_close(raw_changes=None, **changes):
    return {
        "id": "resting",
        "symbol": BTC,
        "position_side": "long",
        "side": "sell",
        "info": {
            "qty": "8",
            "tradeQty": "2",
            "reduceOnly": True,
            **(raw_changes or {}),
        },
        **changes,
    }


@pytest.fixture
def cohort():
    bot = SimpleNamespace(
        MAX_OPEN_ORDERS=100,
        live_value=lambda key: {"time_in_force": "post_only"}[key],
        _loss_gate_max_drawdown_fraction=lambda: 1.0,
        fetched_positions=[
            position(BTC, "long"),
            position(ETH, "short", id_="position-b"),
        ],
        positions={BTC: {"long": {"size": 10}}, ETH: {"short": {"size": -10}}},
        cca=SimpleNamespace(
            _position_ids={(BTC, "long"): "position-a", (ETH, "short"): "position-b"}
        ),
        fetched_open_orders=[],
    )
    return (
        bot,
        [close(BTC, "long"), close(ETH, "short", id_="sibling")],
        {
            BTC: quote(BTC),
            ETH: quote(ETH),
        },
    )


def select(cohort):
    return select_ordinary_close_wave(*cohort, now_ms=NOW, max_age_ms=AGE)


def test_distinct_close_pair_reserves_actual_open_remainder(cohort):
    bot, orders, _ = cohort
    bot.fetched_open_orders = [venue_close()]
    first, second = select(cohort)
    assert first.order is orders[0] and second.order is orders[1]
    assert (first.position_id, second.position_id) == ("position-a", "position-b")
    assert bot.fetched_open_orders == [venue_close()]


def test_existing_client_id_excludes_wave(cohort):
    bot, orders, _ = cohort
    bot.fetched_open_orders = [venue_close(clientOrderId=orders[1]["custom_id"])]
    assert select(cohort) == ()


@pytest.mark.parametrize(
    "change",
    [
        {"type": "market"},
        {"reduce_only": False},
        {"execution_priority": "risk_critical"},
        {"execution_priority": None},
        {"pb_order_type": "close_unstuck_long"},
        {"pb_order_type": "close_panic_long"},
        {"pb_order_type": "entry_grid_normal_long"},
        {"side": "buy"},
        {"custom_id": "bad id"},
        {"qty": float("nan")},
        {"price": False},
        {"price": 98},
    ],
)
def test_unsupported_leader_falls_back_without_reordering(cohort, change):
    _, orders, _ = cohort
    orders.insert(0, {**orders[0], **change})
    assert select(cohort) == ()


@pytest.mark.parametrize("source", ["position_row", "position_cache", "position_size"])
def test_requires_same_exact_authoritative_position(cohort, source):
    bot, _, _ = cohort
    if source == "position_row":
        bot.fetched_positions[0]["info"].pop("positionId")
    elif source == "position_cache":
        bot.cca._position_ids[(BTC, "long")] = "unconfirmed"
    else:
        bot.positions[BTC]["long"]["size"] = 9.9
    assert select(cohort) == ()


@pytest.mark.parametrize(
    "change",
    [
        {"tradeQty": None},
        {"tradeQty": "1.9"},
        {"tradeQty": "2", "dealAmount": "1.9"},
        {"tradeQty": "0"},
        {"reduceOnly": None},
        {"positionId": "other-position"},
    ],
)
def test_unproven_or_oversubscribed_position_capacity_defers_wave(cohort, change):
    bot, _, _ = cohort
    bot.fetched_open_orders = [venue_close(change)]
    assert select(cohort) == ()


def test_exact_decimal_reserve_handles_fractional_qty_without_tolerance(cohort):
    bot, orders, _ = cohort
    bot.fetched_positions[0]["info"]["qty"] = "0.3"
    bot.positions[BTC]["long"]["size"] = 0.3
    orders[0]["qty"] = 0.1
    bot.fetched_open_orders = [venue_close({"qty": "0.2", "tradeQty": "0"})]
    assert len(select(cohort)) == 2
    bot.fetched_open_orders[0]["info"]["qty"] = "0.20000000000000001"
    assert select(cohort) == ()


def test_requires_distinct_symbol_position_and_client_id(cohort):
    bot, orders, _ = cohort
    orders[1]["custom_id"] = orders[0]["custom_id"]
    assert select(cohort) == ()
    orders[1]["custom_id"] = "another"
    bot.fetched_positions[1]["info"]["positionId"] = "position-a"
    bot.cca._position_ids[(ETH, "short")] = "position-a"
    assert select(cohort) == ()
    bot.fetched_positions[1] = position(BTC, "short", id_="position-b")
    bot.positions[BTC]["short"] = {"size": -10}
    bot.cca._position_ids[(BTC, "short")] = "position-b"
    orders[1]["symbol"] = BTC
    assert select(cohort) == ()


def test_requires_capacity_after_all_actual_open_orders(cohort):
    bot, _, _ = cohort
    bot.MAX_OPEN_ORDERS = 3
    bot.fetched_open_orders = [venue_close()]
    assert len(select(cohort)) == 2
    bot.fetched_open_orders.append(venue_close(id="other", symbol="SOL/USDT:USDT"))
    assert select(cohort) == ()
    bot.MAX_OPEN_ORDERS = 4
    assert len(select(cohort)) == 2
    bot.fetched_open_orders[-1]["id"] = "resting"
    assert select(cohort) == ()


def test_loss_gate_or_non_post_only_requires_serial_admission(cohort):
    bot, _, _ = cohort
    bot._loss_gate_max_drawdown_fraction = lambda: 0.99
    assert select(cohort) == ()
    bot._loss_gate_max_drawdown_fraction = lambda: 1.0
    bot.live_value = lambda key: "good_till_cancelled"
    assert select(cohort) == ()


@pytest.mark.parametrize(
    "change",
    [
        {"exchange_timestamp_ms": None},
        {"exchange_timestamp_ms": NOW - AGE - 1},
        {"exchange_timestamp_ms": NOW + 1},
        {"fetched_ms": NOW - AGE - 1},
        {"source": "completed_candle"},
        {"symbol": ETH},
        {"bid": 101},
    ],
)
def test_recheck_only_accepts_a_current_factual_non_crossing_quote(cohort, change):
    first, _ = select(cohort)
    assert not close_is_passive(
        first, replace(quote(BTC), **change), now_ms=NOW, max_age_ms=AGE
    )


def test_wire_must_match_original_economics_and_hedge_side(cohort):
    first, second = select(cohort)
    body = {
        "symbol": "BTCUSDT",
        "clientId": "client-id",
        "positionId": "position-a",
        "qty": "4.0",
        "price": "101.00",
        "side": "BUY",
        "tradeSide": "CLOSE",
        "reduceOnly": True,
        "orderType": "LIMIT",
        "effect": "POST_ONLY",
    }
    assert first.matches_wire(body, venue_symbol="BTCUSDT")
    assert not second.matches_wire(body, venue_symbol="ETHUSDT")
    for key, bad in (
        ("side", "SELL"),
        ("symbol", "ETHUSDT"),
        ("reduceOnly", 1),
        ("price", "101.01"),
        ("qty", "4.001"),
        ("tradeSide", "OPEN"),
        ("positionId", "another"),
        ("orderType", "MARKET"),
        ("effect", "GTC"),
    ):
        assert not first.matches_wire({**body, key: bad}, venue_symbol="BTCUSDT")
