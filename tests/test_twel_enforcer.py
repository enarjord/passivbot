import pytest

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover - executed when the extension is unavailable
    pbr = None

pbr_is_stub = bool(getattr(pbr, "__is_stub__", False)) if pbr is not None else False


def _apply_twel_orders(balance, positions, actions):
    positions_map = {
        pos["idx"]: {
            "size": pos["position_size"],
            "price": pos["position_price"],
            "c_mult": pos["c_mult"],
        }
        for pos in positions
    }
    for idx, qty, _price, _order_type in actions:
        positions_map[idx]["size"] += qty
    total_we = 0.0
    for pos in positions_map.values():
        abs_size = abs(pos["size"])
        if abs_size <= 0.0:
            continue
        total_we += pbr.calc_wallet_exposure(pos["c_mult"], balance, abs_size, pos["price"])
    return total_we


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_twel_reduces_most_profitable_long_first():
    balance = 1000.0
    total_wel = 0.8
    threshold = 1.0
    order_type = pbr.order_type_snake_to_id("close_auto_reduce_twel_long")

    positions = [
        {
            "idx": 0,
            "position_size": 6.0,
            "position_price": 100.0,
            "market_price": 110.0,
            "base_wallet_exposure_limit": 0.2,
            "c_mult": 1.0,
            "qty_step": 0.1,
            "price_step": 0.1,
            "min_qty": 0.0,
        },
        {
            "idx": 1,
            "position_size": 4.0,
            "position_price": 100.0,
            "market_price": 95.0,
            "base_wallet_exposure_limit": 0.2,
            "c_mult": 1.0,
            "qty_step": 0.1,
            "price_step": 0.1,
            "min_qty": 0.0,
        },
    ]

    actions = pbr.calc_twel_enforcer_orders_py(
        "long", threshold, total_wel, 2, balance, positions, None
    )
    assert actions, "Expected TWEL enforcer to emit at least one reduction order"

    first_idx, qty, price, order_type_id = actions[0]
    assert first_idx == 0, "Profitable position should be reduced first"
    assert qty < 0.0
    assert order_type_id == order_type
    assert price <= positions[0]["market_price"] + 1e-9

    final_we = _apply_twel_orders(balance, positions, actions)
    assert final_we <= total_wel * threshold + 1e-9


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_twel_reduces_most_profitable_short_first():
    balance = 1000.0
    total_wel = 0.8
    threshold = 1.0
    order_type = pbr.order_type_snake_to_id("close_auto_reduce_twel_short")

    positions = [
        {
            "idx": 0,
            "position_size": -6.0,
            "position_price": 100.0,
            "market_price": 90.0,
            "base_wallet_exposure_limit": 0.2,
            "c_mult": 1.0,
            "qty_step": 0.1,
            "price_step": 0.1,
            "min_qty": 0.0,
        },
        {
            "idx": 1,
            "position_size": -4.0,
            "position_price": 100.0,
            "market_price": 105.0,
            "base_wallet_exposure_limit": 0.2,
            "c_mult": 1.0,
            "qty_step": 0.1,
            "price_step": 0.1,
            "min_qty": 0.0,
        },
    ]

    actions = pbr.calc_twel_enforcer_orders_py(
        "short", threshold, total_wel, 2, balance, positions, None
    )
    assert actions, "Expected TWEL enforcer to emit at least one reduction order"

    first_idx, qty, price, order_type_id = actions[0]
    assert first_idx == 0, "Profitable short position should be reduced first"
    assert qty > 0.0
    assert order_type_id == order_type
    assert price >= positions[0]["market_price"] - 1e-9

    final_we = _apply_twel_orders(balance, positions, actions)
    assert final_we <= total_wel * threshold + 1e-9
