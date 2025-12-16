import math

import pytest

import passivbot_rust as pbr


def make_asset(idx, bid=10.0, ask=10.0, vol=1_000_000.0, vola=0.1, min_cost=1.0):
    return {
        "idx": idx,
        "bid": bid,
        "ask": ask,
        "volume_score": vol,
        "volatility_score": vola,
        "min_cost": min_cost,
        "min_qty": 0.01,
        "qty_step": 0.01,
        "price_step": 0.01,
        "c_mult": 1.0,
    }


def test_underhedged_adds_shorts():
    res = pbr.compute_hedge_orders_py(
        mode="hedge_shorts_for_longs",
        base_positions=[{"idx": 0, "size": 10.0, "price": 10.0}],
        hedge_positions=[],
        balance=100.0,
        twel=1.5,
        eligible_assets=[make_asset(1), make_asset(2)],
        desired_base_orders=[],
        one_way=False,
    )
    assert res["orders"], "expected hedge orders to be proposed"
    assert any(o["qty"] < 0 for o in res["orders"])


def test_overhedged_reduces_shorts():
    res = pbr.compute_hedge_orders_py(
        mode="hedge_shorts_for_longs",
        base_positions=[{"idx": 0, "size": 5.0, "price": 10.0}],
        hedge_positions=[{"idx": 1, "size": -10.0, "price": 10.0}],
        balance=100.0,
        twel=1.5,
        eligible_assets=[make_asset(1)],
        desired_base_orders=[],
        one_way=False,
    )
    assert any(o["qty"] > 0 for o in res["orders"])


def test_collision_defers_long_and_closes_short():
    res = pbr.compute_hedge_orders_py(
        mode="hedge_shorts_for_longs",
        base_positions=[{"idx": 0, "size": 10.0, "price": 10.0}],
        hedge_positions=[{"idx": 0, "size": -1.0, "price": 10.0}],
        balance=100.0,
        twel=1.5,
        eligible_assets=[make_asset(0)],
        desired_base_orders=[{"idx": 0}],
        one_way=True,
    )
    assert res["deferred_base_longs"] == [0]
    assert any(o["action"] == "close" for o in res["orders"])


def test_short_only_mode_adds_longs():
    res = pbr.compute_hedge_orders_py(
        mode="hedge_longs_for_shorts",
        base_positions=[{"idx": 0, "size": -10.0, "price": 10.0}],
        hedge_positions=[],
        balance=100.0,
        twel=1.5,
        eligible_assets=[make_asset(1)],
        desired_base_orders=[],
        one_way=False,
    )
    assert any(o["qty"] > 0 for o in res["orders"])
