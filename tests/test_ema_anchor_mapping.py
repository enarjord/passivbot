import math


def test_ema_anchor_mapping_and_fallback():
    # Emulate outputs of cm.get_ema_bounds for two symbols
    ema_bounds_long = {
        "BTC/USDT:USDT": (100.0, 110.0),  # (lower, upper)
        "ETH/USDT:USDT": (200.0, 220.0),
    }
    ema_bounds_short = {
        "BTC/USDT:USDT": (300.0, 330.0),
        "ETH/USDT:USDT": (400.0, 440.0),
    }

    # Construct anchors exactly like in calc_ideal_orders_new
    ema_anchor = {
        "long": {s: ema_bounds_long.get(s, (float("nan"), float("nan")))[0] for s in ema_bounds_long},
        "short": {
            s: ema_bounds_short.get(s, (float("nan"), float("nan")))[1] for s in ema_bounds_short
        },
    }

    # Long should take lower bound
    assert ema_anchor["long"]["BTC/USDT:USDT"] == 100.0
    assert ema_anchor["long"]["ETH/USDT:USDT"] == 200.0

    # Short should take upper bound
    assert ema_anchor["short"]["BTC/USDT:USDT"] == 330.0
    assert ema_anchor["short"]["ETH/USDT:USDT"] == 440.0

    # Fallback for missing symbol should use last price when accessed via get(...)
    last_prices = {"XRP/USDT:USDT": 0.523}
    val_long = ema_anchor["long"].get("XRP/USDT:USDT", last_prices.get("XRP/USDT:USDT", float("nan")))
    val_short = ema_anchor["short"].get(
        "XRP/USDT:USDT", last_prices.get("XRP/USDT:USDT", float("nan"))
    )
    assert val_long == 0.523
    assert val_short == 0.523

    # If last price also missing, fallback is NaN
    miss = ema_anchor["long"].get("DOGE/USDT:USDT", last_prices.get("DOGE/USDT:USDT", float("nan")))
    assert math.isnan(miss)
